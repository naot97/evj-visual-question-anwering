import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from torchvision.transforms import InterpolationMode

    
from dataset.my_pretrain_dataset_multilingual import ImageMultiTextDataset, ImageMonoTextDataset

from dataset.retrieval_dataset import re_train_dataset, re_eval_dataset
from dataset.nlvr_dataset import nlvr_dataset
from dataset.vqa_dataset import vqa_dataset

from dataset.xvnli_dataset import xvnli_dataset
from dataset.xflickrco_dataset import xflickrco_train_dataset, xflickrco_eval_dataset
from dataset.wit_dataset import wit_train_dataset, wit_eval_dataset
from dataset.uit_vqa_dataset import uit_vqa_dataset
from dataset.randaugment import RandomAugment




def create_dataset(dataset, config, tokenizer):
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    pretrain_transform = transforms.Compose([   
        transforms.RandomResizedCrop(config['image_res'], scale=(0.2, 1.0),
                                     interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
        transforms.ToTensor(),
        normalize,
    ])

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(config['image_res'], scale=(0.5, 1.0),
                                     interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
        transforms.ToTensor(),
        normalize,
    ])

    train_transform_wohflip = transforms.Compose([
        transforms.RandomResizedCrop(config['image_res'], scale=(0.5, 1.0),
                                     interpolation=InterpolationMode.BICUBIC),
        # transforms.RandomHorizontalFlip(),
        RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
        transforms.ToTensor(),
        normalize,
    ])

    box_transform = transforms.Compose([
        RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness']),
        transforms.ToTensor(),
        normalize,
    ])

    test_transform = transforms.Compose([
        transforms.Resize((config['image_res'], config['image_res']), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ])

    if dataset == 'pretrain_multilingual':
        general_dataset = ImageMultiTextDataset(config, config['train_file'], tokenizer, shuffle=True,
                                                repeat=True, transform=pretrain_transform)

        region_dataset = None

        mono_dataset = ImageMonoTextDataset(config, config['train_file_mono'], tokenizer, shuffle=True,
                                            repeat=True, transform=pretrain_transform)

        # if len(config['train_file_text']):
        #     text_dataset = ParaTextDataset(config, config['train_file_text'], tokenizer, rank=int(os.environ.get('RANK') or 0),
        #                                    world_size=int(os.environ.get('WORLD_SIZE') or 1), shuffle=True, repeat=True)
        # else:
        #     text_dataset = None
        text_dataset = None

        return general_dataset, region_dataset, mono_dataset, text_dataset

    elif dataset == 'gqa':
        train_dataset = vqa_dataset(config['train_file'], train_transform_wohflip, config['vqa_root'],
                                    split='train', text_encoder=config['text_encoder'])

        valid_dataset = vqa_dataset(config['valid_file'], test_transform, config['vqa_root'],
                                    split='test', answer_list=config['answer_list'],
                                    text_encoder=config['text_encoder'])

        test_dataset_dict = {}
        for language, (rpath, ans_rpath) in config['test_file'].items():
            test_dataset_dict[language] = vqa_dataset(rpath, test_transform, config['vqa_root'], split='test', answer_list=ans_rpath,
                                                      text_encoder=config['text_encoder'])

        return train_dataset, valid_dataset, test_dataset_dict
    elif dataset == 'uit':
        train_dataset = uit_vqa_dataset(config['train_file'], train_transform_wohflip, config['vqa_root'],
                                    split='train', tokenizer = tokenizer)

        valid_dataset = uit_vqa_dataset(config['valid_file'], test_transform, config['vqa_root'],
                                    split='valid',
                                    tokenizer = tokenizer)
        test_dataset =  uit_vqa_dataset(config['test_file'], test_transform, config['test_root'],
                                    split='test', tokenizer = tokenizer)
        return train_dataset, valid_dataset, test_dataset 
    
    else:
        raise NotImplementedError(f"dataset == {dataset}")


def vqa_collate_fn(batch):
    image_list, question_list, answer_list, weight_list, n = [], [], [], [], []
    for image, question, answer, weights in batch:
        image_list.append(image)
        question_list.append(question)
        weight_list += weights       
        answer_list += answer
        n.append(len(answer))
    return torch.stack(image_list, dim=0), question_list, answer_list, torch.Tensor(weight_list), n


def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset, shuffle in zip(datasets, shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)
        samplers.append(sampler)
    return samplers


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset, sampler, bs, n_worker, is_train, collate_fn in zip(datasets, samplers, batch_size, num_workers,
                                                                    is_trains, collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )
        loaders.append(loader)

    if len(loaders) <= 1:
        print(f"### be careful: func create_loader returns a list length of {len(loaders)}")

    return loaders
