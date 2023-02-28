#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Multi-Grained Vision Language Pre-Training: Aligning Texts with Visual Concepts (https://arxiv.org/abs/2111.08276)
# Github: https://github.com/zengyan-97/X-VLM
# Copyright (c) 2022, ByteDance Inc.
# All rights reserved.

import sys
from typing import List, Any
import warnings
import random
from itertools import cycle
import torch
from torch.utils.data import Dataset
import copy
import json


my_data = {
    'en': ['pre_train/annotations/captions_train2017.json'],
    'vi': ['pre_train/UIT-ViIC/uitviic_captions_train2017.json'],
    'ja': ['pre_train/STAIR-captions/stair_captions_v1.2_train.json']
}

class DistLineReadingDataset(Dataset):  # pylint: disable=W0223
    """
    iterate a set of folders.
    """
    def __init__(self,
                 ann_file: str,
                images_root = '/content/drive/MyDrive/train2017/'):
        super().__init__()
        self.images_root = images_root
        self.ann = {}
        self.images = {}

        for language, file_lst in my_data.items():
            for f in file_lst:
                data = json.load(open(f, 'r'))  
                images = data['images']
                for image in images:
                    self.images[image['id']] = image

                for ann in data['annotations']:
                    if ann['image_id'] in self.images.keys():
                        if ann['image_id'] not in self.ann.keys():
                            self.ann[ann['image_id']] = {language: [ann['caption']]}
                        elif language not in self.ann[ann['image_id']].keys():
                            self.ann[ann['image_id']][language] = [ann['caption']] 
                        else:
                            self.ann[ann['image_id']][language].append(ann['caption'])

        temp_keys = copy.copy(list(self.ann.keys()))
        for image_id in temp_keys:
            if len(self.ann[image_id].keys()) < 2:
                del self.ann[image_id]

        with open('data.json', 'w', encoding='utf-8') as f:
            json.dump(self.ann, f, ensure_ascii=False, indent=4)

    def __len__(self):
        return len(self.ann)



def split_shard(data: List[Any], shard_idx: int, shard_size: int):
    num = len(data)
    if num < shard_size:
        raise RuntimeError("num:{} < shard size:{}".format(num, shard_size))
    start_idx = (num * shard_idx) // shard_size
    end_idx = (num * (shard_idx + 1)) // shard_size
    return data[start_idx: end_idx]
