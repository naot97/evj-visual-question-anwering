import os
import json
import random
from random import random as rand

from PIL import Image
from torch.utils.data import Dataset
from dataset.utils import pre_question

from torchvision.transforms.functional import hflip

class uit_vqa_dataset(Dataset):
    def __init__(self, ann_file, transform, vqa_root=None, split="train", max_ques_words=30, answer_list='',
                 tokenizer = None):

        self.careful_hflip = True

        self.split = split
        self.ann = []
        self.images = {}

        if isinstance(ann_file, str):
            ann_file = [ann_file]
        elif not isinstance(ann_file, list):
            raise ValueError

        
        for f in ann_file:
            data = json.load(open(f, 'r'))
            images = data['images']
            for image in images:
                self.images[image['id']] = image

            for ann in data['annotations']:
                if ann['image_id'] in self.images.keys():
                    self.ann.append(ann)       

        self.transform = transform
        self.vqa_root = vqa_root
        self.max_ques_words = max_ques_words

        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token = tokenizer.sep_token

        if split == 'valid':
            self.max_ques_words = 50  # do not limit question length during test
            self.ann = [self.ann[i] for i in range(len(self.ann)) if i % 10 in [8,9] ]
        elif split == 'train':
            self.ann = [self.ann[i] for i in range(len(self.ann)) if i % 10 not in [8,9] ]
        elif split == 'test':
            self.max_ques_words = 50

        
    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        ann = self.ann[index]

        image_path = os.path.join(self.vqa_root, self.images[ann['image_id']]['filename'])

        image = Image.open(image_path).convert('RGB')

        image = self.transform(image)

        if self.split in ['test', 'valid']:
            question = pre_question(ann['question'], self.max_ques_words)
            question_id = ann['id']

            return image, question, question_id

        elif self.split == 'train':
            question = pre_question(ann['question'], self.max_ques_words)

            answers = [ann['answer']]
            weights = [1]
            return image, question, answers, weights

        else:
            raise NotImplementedError
