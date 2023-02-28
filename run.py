import os
import sys
import time
import random
import argparse

# sys.path.append('/content/drive/MyDrive/CCLM')
from utils.hdfs_io import HADOOP_BIN, hexists, hmkdir, hcopy
from utils.marvl_preproc import marvl_preproc
from utils.wit_preproc import wit_preproc
# os.environ['MKL_THREADING_LAYER'] = 'GNU'
############ Set it correctly for distributed training across nodes


def run_pretrain(args):
    print("### Start pre-training", flush=True)
    os.system(f"python3 my_Pretrain.py "
              f"--config {args.config}")



def run_vqa(args, load_vqa_pretrain=False):
    # dist_launch = get_dist_launch(args)

    #assert os.path.exists("images/gqa")

    print("### Training VQA", flush=True)
    args.config = f"configs/{args.model}/GQA_fewshot.yaml" if args.fewshot else f"configs/{args.model}/GQA.yaml"

    os.system(
              f"python3 VQA.py --config {args.config} {'--load_vqa_pretrain' if args.load_vqa_pretrain else ''}"
              f" --output_dir {args.output_dir} "
              f"--bs {args.bs} {'--evaluate' if args.evaluate else ''} "
              f"{'--checkpoint ' + args.checkpoint if args.evaluate or args.load_vqa_pretrain  else ''}"
              f"{'--load_vqa_pretrain --fewshot ' + args.fewshot if args.fewshot else ''} ")



def run(args):
    if args.task == 'pretrain_cclm_3m':
        args.config = 'configs/my_Pretrain_3m.yaml'
        run_pretrain(args)

    elif args.task == 'pretrain_cclm_4m':
        args.config = 'configs/Pretrain_4m.yaml'
        run_pretrain(args)


    elif args.task == 'gqa':
        run_vqa(args)


    else:
        raise NotImplementedError(f"task == {args.task}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True)
    # parser.add_argument('--dist', type=str, required=True, help="see func get_dist_launch for details")

    parser.add_argument('--config', default='', type=str, help="if not given, use default")
    parser.add_argument('--model', default='cclm-base-ft', type=str, help="to set default fine-tuning configs")

    parser.add_argument('--epoch', default=-1, type=int, help="for pre-training (debug) only")
    parser.add_argument('--bs', default=-1, type=int, help="for each gpu, batch_size = bs // num_gpus; "
                                                           "this option only works for fine-tuning scripts.")

    parser.add_argument('--checkpoint', default='', type=str, help="for fine-tuning")
    parser.add_argument('--load_ckpt_from', default='', type=str, help="load domain pre-trained params")

    # write path: local or HDFS
    parser.add_argument('--output_dir', type=str, required=True, help='for fine-tuning, local path; '
                                                                      'for pre-training, local and HDFS are both allowed.')
    parser.add_argument('--output_hdfs', type=str, default='', help="HDFS path required by VQA and Refcoco, "
                                                                    "to collect eval results among nodes")

    parser.add_argument('--evaluate', action='store_true', help="evaluation on downstream tasks")
    parser.add_argument('--load_vqa_pretrain', action='store_true', help="evaluation on downstream tasks")
    parser.add_argument('--seed', default=42, type=int)

    parser.add_argument('--fewshot', default='', type=str, help="IGLUE fewshot. <lang>,<shot_num>, eg: ar,25")
    parser.add_argument('--lr', default=0., type=float, help="learning rate")
    parser.add_argument('--gmt', action='store_true', help="whether use google machine translation as test set")

    args = parser.parse_args()
    print(args.output_dir)
    
    hmkdir(args.output_dir)

    if len(args.config):
        assert hexists(args.config)

        if args.config.startswith('hdfs://'):
            args.config = get_from_hdfs(args.config)

    if args.checkpoint.startswith('hdfs://'):
        args.checkpoint = get_from_hdfs(args.checkpoint)

    run(args)
