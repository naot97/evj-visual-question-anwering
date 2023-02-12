import argparse
import os
import math
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from models.my_model_vqa import VQAModel

import utils
from utils.checkpointer import Checkpointer

from dataset.utils import collect_result
from dataset import create_dataset, create_sampler, create_loader, vqa_collate_fn

from scheduler import create_scheduler
from optim import create_optimizer

from transformers import AutoTokenizer

def build_tokenizer(text_encoder: str):
    tokenizer = AutoTokenizer.from_pretrained(text_encoder)
    tokenizer.add_special_tokens({'bos_token': tokenizer.cls_token, 'eos_token': tokenizer.sep_token})
    return tokenizer

def train(model, data_loader, optimizer, encoder_tokenizer,decoder_tokenizer, epoch, device, scheduler, config):
    header = 'Train Epoch: [{}]'.format(epoch)
    model.train()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

    print_freq = 50

    for i, (image, question, answer, weights, n) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image = image.to(device, non_blocking=True)
        question_input = encoder_tokenizer(question, padding='longest', truncation=True, max_length=config['max_tokens'], return_tensors="pt").to(device)
        answer_input = decoder_tokenizer(answer, padding='longest', return_tensors="pt").to(device) 
        loss = model(image, question_input, answer_input, train=True, k=n, weights=weights)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.5f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluation(model, data_loader, tokenizer, device, config) :
    # test
    model.eval()
            
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Generate VQA test result:'
    print_freq = 50
    
    result = []

    for n, (image, question, question_id) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):        
        image = image.to(device, non_blocking=True)
        question_input = tokenizer(question, padding='longest', truncation=True, max_length=config['max_tokens'], return_tensors="pt").to(device)        

        outs = model(image, question_input, train=False)      
        for i, out in enumerate(outs):
          result.append({"question_id": question_id[i], "answer": out})


    print(outs[:4])
    print(question[:4])
    return result 

@torch.no_grad()
def submit(model, data_loader, tokenizer, device, config) :
    # test
    model.eval()
            
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Generate VQA test result:'
    print_freq = 50
    
    result = {}

    for n, (image, question, question_id) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):        
        image = image.to(device, non_blocking=True)
        question_input = tokenizer(question, padding='longest', return_tensors="pt").to(device)        

        outs = model(image, question_input,  train=False)
        for i, out in enumerate(outs):
          result.update({question_id[i].cpu().detach().item(): out})

    return result

def get_acc(results, test_file, ann):
    # run eval
    preds = {}
    for pred in results:
        preds[int(pred['question_id'])] = pred['answer']
    # test_data = []
    n, n_correct = 0, 0
    for sample in ann:
        if 'answer' in sample.keys():
            n += 1
            if int(sample['id']) in preds and preds[int(sample['id'])] == sample['answer']:
                n_correct += 1

    print(f"n: {n}, n_correct: {n_correct}, acc: {n_correct / n}", flush=True)
    return n_correct / n if n > 0 else 0


def main(args, config):
    # device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu") 
    device = torch.device("cpu") 

    seed = args.seed 
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    
    start_epoch = 0
    max_epoch = config['schedular']['epochs']

    print("Creating vqa datasets")
    decoder_tokenizer = build_tokenizer(config['text_decoder'])
    encoder_tokenizer = build_tokenizer(config['text_encoder'])
    train_dataset, valid_dataset, test_dataset = create_dataset('uit', config, decoder_tokenizer)
    datasets = [train_dataset, valid_dataset]

    train_dataset_size = len(train_dataset)

    samplers = [None, None]

    train_loader, valid_loader = create_loader(datasets, samplers,
                                              batch_size=[args.bs, args.bs],
                                              num_workers=[4, 4], is_trains=[True, False],
                                              collate_fns=[vqa_collate_fn, None])

    test_loader = create_loader([test_dataset], [None], batch_size=[args.bs],
                                        num_workers=[4], is_trains=[False], collate_fns=[None])[0]

    print("Creating model")
    print("### pad_token_id, ", train_dataset.pad_token_id)
    print("### eos_token, ", train_dataset.eos_token)
    model = VQAModel(config=config, tokenizer = decoder_tokenizer)
    if args.load_vqa_pretrain:
        model.load_pretrained(args.checkpoint, config)
    print(device)
    print("### Total Params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    start_time = time.time()
    print("### output_dir, ", args.output_dir, flush=True)
    model = model.to(device)

    if args.evaluate:
        model.load_pretrained(args.checkpoint, config, is_eval=args.evaluate or args.load_vqa_pretrain)
        results =  submit(model, test_loader, tokenizer, device, config)
        with open('results.json', 'w', encoding='utf-8') as f:
          json.dump(results, f, ensure_ascii=False, indent=4)
    else:
        print("Start training")
        arg_opt = utils.AttrDict(config['optimizer'])
        optimizer = create_optimizer(arg_opt, model)
        arg_sche = utils.AttrDict(config['schedular'])
        arg_sche['step_per_epoch'] = math.ceil(train_dataset_size / args.bs)
        lr_scheduler = create_scheduler(arg_sche, optimizer)

        checkpointer = Checkpointer(args.output_dir)

        best = -1
        best_epoch = 0
        if 'eval_interval' not in config:
            config['eval_interval'] = 1
        print("STARTING...")
        for epoch in range(start_epoch, max_epoch):
            print(epoch)
            train_stats = train(model, train_loader, optimizer, encoder_tokenizer, decoder_tokenizer, epoch, device, lr_scheduler, config)

            if epoch >= config['start_eval']:
                vqa_result = evaluation(model, valid_loader, encoder_tokenizer, device, config)
                result = vqa_result

                print("Evaluating on valid set", flush=True)
                valid_acc = get_acc(result, config['valid_file'], valid_dataset.ann)

                print(valid_acc)
                if valid_acc > best or epoch == max_epoch - 1 or epoch % 5 == 0:
                    best = valid_acc
                    best_epoch = epoch

                    model_without_ddp = model
                    if hasattr(model, 'module'):
                        model_without_ddp = model.module

                    save_obj = {
                        'model': model_without_ddp.state_dict(),
                        'config': config,
                    }
                    checkpointer.save_checkpoint(model_state=save_obj,
                                                epoch=epoch,
                                                training_states=optimizer.state_dict())

                print("### best_epoch, ", best_epoch, flush=True)
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                                 'valid_acc': valid_acc,
                                 'epoch': epoch}
                with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                        f.write(json.dumps(log_stats) + "\n")
                
        os.system(f"cat {args.output_dir}/log.txt")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('### Time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--config', default='./configs/cclm-base-ft/GQA.yaml')
    parser.add_argument('--output_dir', default='output/vqa')

    parser.add_argument('--bs', default=-1, type=int)
    parser.add_argument('--evaluate', default=False,action='store_true')
    parser.add_argument('--load_vqa_pretrain', action='store_true')

    args = parser.parse_args()
    args.seed = 42
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    args.result_dir = os.path.join(args.output_dir, 'result')
    os.makedirs(args.result_dir, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))
    main(args, config)
