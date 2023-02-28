import json
import copy
import math
import random
import sys
import re
import io
import traceback
from base64 import b64decode
import os
from random import randint, shuffle
from random import random as rand

import torch
from torchvision.transforms.functional import hflip, resize
from torchvision.transforms import InterpolationMode

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from dataset.utils import pre_caption
from dataset.my_dist_dataset import DistLineReadingDataset
from torch.utils.data import Dataset

class TextMaskingGenerator:
    def __init__(self, tokenizer, mask_prob, mask_max, skipgram_prb=0.2, skipgram_size=3, mask_whole_word=True, use_roberta=False):
        self.id2token = {i: w for w, i in tokenizer.get_vocab().items()}
        print("len(tokenizer.id2token), ", len(self.id2token), flush=True)

        self.use_roberta = use_roberta

        for i in range(len(self.id2token)):
            assert i in self.id2token.keys()  # check

        self.cls_token = tokenizer.cls_token
        self.mask_token = tokenizer.mask_token

        self.mask_max = mask_max
        self.mask_prob = mask_prob

        self.skipgram_prb = skipgram_prb
        self.skipgram_size = skipgram_size
        self.mask_whole_word = mask_whole_word

    def get_random_word(self):
        i = randint(0, len(self.id2token) - 1)
        return self.id2token[i]

    def __call__(self, tokens: list):  # tokens: [CLS] + ...
        n_pred = min(self.mask_max, max(
            1, int(round(len(tokens) * self.mask_prob))))

        # candidate positions of masked tokens
        assert tokens[0] == self.cls_token
        special_pos = set([0])  # will not be masked
        cand_pos = list(range(1, len(tokens)))

        shuffle(cand_pos)
        masked_pos = set()
        max_cand_pos = max(cand_pos)
        for pos in cand_pos:
            if len(masked_pos) >= n_pred:
                break
            if pos in masked_pos:
                continue

            def _expand_whole_word(st, end):
                new_st, new_end = st, end

                if self.use_roberta:
                    while (new_st > 1) and (tokens[new_st][0] != 'Ġ'):
                        new_st -= 1
                    while (new_end < len(tokens)) and (tokens[new_end][0] != 'Ġ'):
                        new_end += 1
                else:
                    # bert, WordPiece
                    while (new_st >= 0) and tokens[new_st].startswith('##'):
                        new_st -= 1
                    while (new_end < len(tokens)) and tokens[new_end].startswith('##'):
                        new_end += 1

                return new_st, new_end

            if (self.skipgram_prb > 0) and (self.skipgram_size >= 2) and (rand() < self.skipgram_prb):
                # ngram
                cur_skipgram_size = randint(2, self.skipgram_size)
                if self.mask_whole_word:
                    st_pos, end_pos = _expand_whole_word(
                        pos, pos + cur_skipgram_size)
                else:
                    st_pos, end_pos = pos, pos + cur_skipgram_size
            else:
                if self.mask_whole_word:
                    st_pos, end_pos = _expand_whole_word(pos, pos + 1)
                else:
                    st_pos, end_pos = pos, pos + 1

            for mp in range(st_pos, end_pos):
                if (0 < mp <= max_cand_pos) and (mp not in special_pos):
                    masked_pos.add(mp)
                else:
                    break

        masked_pos = list(masked_pos)
        n_real_pred = len(masked_pos)
        if n_real_pred > n_pred:
            shuffle(masked_pos)
            masked_pos = masked_pos[:n_pred]

        for pos in masked_pos:
            if rand() < 0.8:  # 80%
                tokens[pos] = self.mask_token
            elif rand() < 0.5:  # 10%
                tokens[pos] = self.get_random_word()

        return tokens, masked_pos   

class ImageMultiTextDataset(DistLineReadingDataset):
    def __init__(self, config, ann_file, tokenizer, shuffle=True, repeat=True, transform=None):
        super().__init__(ann_file,config['image_root'])

        if 'images' in config.keys():
            self.image_key = config['images']['image_key']
            self.is_image_rpath = config['images']['is_image_rpath']
            self.caption_key = config['images']['caption_key']
            self.batch_size = config['images']['batch_size']
            self.tokenized = config['images']['tokenized']
            if 'language_chosen' in config['images'].keys():
                assert isinstance(config['images']['language_chosen'], list)
                self.language_chosen = set(config['images']['language_chosen'])
                print("### language_chosen, ", self.language_chosen, flush=True)
            else:
                self.language_chosen = set() 

        self.tokenizer = tokenizer

        self.add_eos = True  # always add eos

        self.cls_token = self.tokenizer.cls_token
        self.eos_token = self.tokenizer.sep_token
        self.pad_token_id = self.tokenizer.pad_token_id
        self.mask_token_id = self.tokenizer.mask_token_id

        self.mask_generator = TextMaskingGenerator(self.tokenizer, config['mask_prob'],
                                                   config['max_masks'], config['skipgram_prb'],
                                                   config['skipgram_size'], mask_whole_word=False)

        self.PAD_mask = -100  # loss will ignore this
        self.max_words = config['max_words']
        self.max_tokens = config['max_tokens']
        self.max_masks = config['max_masks']

        self.transform = transform
        self.image_res = config['image_res']
        self.patch_size = config['patch_size']
        assert self.image_res % self.patch_size == 0
        self.num_patch = int(self.image_res / self.patch_size)

        self.sample_2_captions = config['sample_2_captions']
        print("### sample_2_captions: ", self.sample_2_captions)

    def get_caption(self, captions, return_keys=False):
        captions = captions[random.choice(list(captions.keys()))]
        caption = random.choice(captions)
        return caption

    def get_2_captions(self, captions, return_keys=False):
        languages = random.sample(list(captions.keys()), 2)
        return random.choice(captions[languages[0]]), random.choice(captions[languages[1]])
          

    def __getitem__(self, index):
        k = list(self.ann.keys())[index]
        ann = self.ann[k]

        image_path = os.path.join(self.images_root, self.images[k]['file_name'])
        image_path = image_path.replace('train2014','train2017')
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        if self.sample_2_captions and len(ann) >= 2:
            caption, caption_2 = self.get_2_captions(ann)

            text_ids, text_atts, text_ids_masked, masked_pos, masked_ids = self.preprocess(caption)
            text_ids_2, text_atts_2, text_ids_masked_2, masked_pos_2, masked_ids_2 = self.preprocess(caption_2)

        else:
            caption = self.get_caption(ann)
            text_ids, text_atts, text_ids_masked, masked_pos, masked_ids = self.preprocess(caption)
            text_ids_2, text_atts_2, text_ids_masked_2, masked_pos_2, masked_ids_2 = [None] * 5

        return image, text_ids, text_atts, text_ids_masked, masked_pos, masked_ids, \
                      text_ids_2, text_atts_2, text_ids_masked_2, masked_pos_2, masked_ids_2

        
    def preprocess(self, text):
        if self.tokenized:
            tokens = text.strip().split(' ')
        else:
            text = pre_caption(text, self.max_words)  # be careful, if text is '', it will cause error
            tokens = self.tokenizer.tokenize(text)

        tokens = [self.cls_token] + tokens[:self.max_tokens - 1]

        if self.add_eos:
            tokens = tokens[:self.max_tokens - 1]
            tokens += [self.eos_token]

        n_tokens = len(tokens)
        assert n_tokens >= 2, "len(word tokens) < 2"

        text_ids = self.tokenizer.convert_tokens_to_ids(tokens)  # list of int

        tokens_masked, masked_pos = self.mask_generator(copy.deepcopy(tokens))
        text_ids_masked = self.tokenizer.convert_tokens_to_ids(tokens_masked)  # list of int
        masked_ids = [text_ids[p] for p in masked_pos]

        # pad
        n_pad = self.max_tokens - n_tokens
        text_ids = text_ids + [self.pad_token_id] * n_pad
        text_atts = [1] * n_tokens + [0] * n_pad

        text_ids_masked = text_ids_masked + [self.pad_token_id] * n_pad
        n_pad = self.max_masks - len(masked_ids)
        masked_pos = masked_pos + [0] * n_pad
        masked_ids = masked_ids + [self.PAD_mask] * n_pad

        return text_ids, text_atts, text_ids_masked, masked_pos, masked_ids

    def collate_fn(self, batch):
        batch_tensors = []
        for x in zip(*batch):
            if x[0] is None:
                batch_tensors.append(None)
            elif isinstance(x[0], torch.Tensor):
                batch_tensors.append(torch.stack(x))
            else:
                batch_tensors.append(torch.tensor(x, dtype=torch.long))

        return batch_tensors


class ImageMonoTextDataset(ImageMultiTextDataset):
    def __init__(self, config, data_path, tokenizer, shuffle=True, repeat=True, transform=None):
        super().__init__(config, data_path, tokenizer, shuffle=shuffle,
                         repeat=repeat, transform=transform)

        self.add_eos = True  # always add eos

        self.image_key = config['images_mono']['image_key']
        self.is_image_rpath = config['images_mono']['is_image_rpath']
        self.caption_key = config['images_mono']['caption_key']
        self.batch_size = config['images_mono']['batch_size']
        self.tokenized = config['images_mono']['tokenized']

    def __getitem__(self, index):

        ann = self.ann[index]

        image_path = os.path.join(self.images_root, self.images[ann['image_id']]['file_name'])
        image_path = image_path.replace('train2014','train2017')
        image = Image.open(image_path).convert('RGB')

        image = self.transform(image)

        caption = self.get_caption(ann[self.caption_key])

        text_ids, text_atts, text_ids_masked, masked_pos, masked_ids = self.preprocess(caption)

        return image, text_ids, text_atts, text_ids_masked, masked_pos, masked_ids


# class ParaTextDataset(DistLineReadingDataset):
#     def __init__(self, config, data_path, tokenizer, rank=0, world_size=1, shuffle=True, repeat=True):
#         super().__init__(data_path)

#         self.source_key = config['texts_para']['source_key']
#         self.target_key = config['texts_para']['target_key']
#         self.tokenized = config['texts_para']['tokenized']
#         if 'language_chosen' in config.keys():
#             assert isinstance(config['texts_para']['language_chosen'], list)
#             self.language_chosen = set(config['texts_para']['language_chosen'])
#         else:
#             self.language_chosen = set()

#         # assert 'xlm-roberta' in config['text_encoder'], "otherwise, not implemented yet"
#         self.tokenizer = tokenizer

#         self.add_eos = True  # always add eos

#         self.cls_token = self.tokenizer.cls_token
#         self.eos_token = self.tokenizer.sep_token
#         self.pad_token_id = self.tokenizer.pad_token_id
#         self.mask_token_id = self.tokenizer.mask_token_id

#         print("dataset.cls_token, ", self.cls_token, flush=True)
#         print("dataset.eos_token, ", self.eos_token, flush=True)
#         print("dataset.pad_token_id, ", self.pad_token_id, flush=True)
#         print("dataset.mask_token_id, ", self.mask_token_id, flush=True)

#         self.use_tlm = True if ('use_tlm' in config) and config['use_tlm'] else False

#         if 'max_words' in config['texts_para']:
#             self.max_words = config['texts_para']['max_words']
#             self.max_tokens = config['texts_para']['max_tokens']
#             self.mask_prob = config['texts_para']['mask_prob']
#             self.max_masks = config['texts_para']['max_masks']
#         else:
#             self.max_words = config['max_words']
#             self.max_tokens = config['max_tokens']
#             self.mask_prob = config['mask_prob']
#             self.max_masks = config['max_masks']

#         self.mask_generator = TextMaskingGenerator(self.tokenizer, self.mask_prob,
#                                                    self.max_masks, config['skipgram_prb'],
#                                                    config['skipgram_size'], mask_whole_word=False)

#         self.PAD_mask = -100  # loss will ignore this


#     def __getitem__(self, index):

#         ann = self.ann[index]


#         if rand() < 0.5:
#             caption, caption_2 = ann[self.source_key], ann[self.target_key]
#         else:
#             caption, caption_2 = ann[self.target_key], ann[self.source_key]

#         if self.use_tlm:
#             text_ids, text_atts = self.preprocess(caption, return_mask=False)
#             text_ids_2, text_atts_2 = self.preprocess(caption_2, return_mask=False)

#             _, text_atts_masked, text_ids_masked, masked_pos, masked_ids = self.preprocess_tlm(caption, caption_2)

#             return text_ids, text_atts, text_ids_2, text_atts_2, \
#                     text_ids_masked, text_atts_masked, masked_pos, masked_ids

#         else:
#             text_ids, text_atts, text_ids_masked, masked_pos, masked_ids = self.preprocess(caption)
#             text_ids_2, text_atts_2, text_ids_masked_2, masked_pos_2, masked_ids_2 = self.preprocess(caption_2)

#             return text_ids, text_atts, text_ids_masked, masked_pos, masked_ids, \
#                     text_ids_2, text_atts_2, text_ids_masked_2, masked_pos_2, masked_ids_2


#     def preprocess(self, text, return_mask=True):
#         if self.tokenized:
#             tokens = text.strip().split(' ')
#         else:
#             text = pre_caption(text, self.max_words)  # be careful, if text is '', it will cause error
#             tokens = self.tokenizer.tokenize(text)

#         tokens = [self.cls_token] + tokens[:self.max_tokens - 1]

#         if self.add_eos:
#             tokens = tokens[:self.max_tokens - 1]
#             tokens += [self.eos_token]

#         n_tokens = len(tokens)
#         assert n_tokens >= 2, "len(word tokens) < 2"

#         text_ids = self.tokenizer.convert_tokens_to_ids(tokens)  # list of int

#         if return_mask:
#             tokens_masked, masked_pos = self.mask_generator(copy.deepcopy(tokens))
#             text_ids_masked = self.tokenizer.convert_tokens_to_ids(tokens_masked)  # list of int
#             masked_ids = [text_ids[p] for p in masked_pos]

#             # pad
#             n_pad = self.max_tokens - n_tokens
#             text_ids = text_ids + [self.pad_token_id] * n_pad
#             text_atts = [1] * n_tokens + [0] * n_pad

#             text_ids_masked = text_ids_masked + [self.pad_token_id] * n_pad
#             n_pad = self.max_masks - len(masked_ids)
#             masked_pos = masked_pos + [0] * n_pad
#             masked_ids = masked_ids + [self.PAD_mask] * n_pad

#             return text_ids, text_atts, text_ids_masked, masked_pos, masked_ids

#         else:
#             # pad
#             n_pad = self.max_tokens - n_tokens
#             text_ids = text_ids + [self.pad_token_id] * n_pad
#             text_atts = [1] * n_tokens + [0] * n_pad

#             return text_ids, text_atts

#     def preprocess_tlm(self, text, text2):
#         if self.tokenized:
#             tokens = text.strip().split(' ')
#             tokens2 = text2.strip().split(' ')

#         else:
#             text = pre_caption(text, self.max_words)  # be careful, if text is '', it will cause error
#             tokens = self.tokenizer.tokenize(text)

#             text2 = pre_caption(text2, self.max_words)  # be careful, if text is '', it will cause error
#             tokens2 = self.tokenizer.tokenize(text2)

#         tokens = tokens[:self.max_tokens-1]
#         tokens2 = tokens2[:self.max_tokens-1]

#         tokens = [self.cls_token] + tokens + [self.tokenizer.eos_token] + tokens2

#         if self.add_eos:
#             tokens = tokens[:2*self.max_tokens - 1]
#             tokens += [self.eos_token]

#         n_tokens = len(tokens)
#         assert n_tokens >= 2, "len(word tokens) < 2"

#         text_ids = self.tokenizer.convert_tokens_to_ids(tokens)  # list of int

#         tokens_masked, masked_pos = self.mask_generator(copy.deepcopy(tokens))
#         text_ids_masked = self.tokenizer.convert_tokens_to_ids(tokens_masked)  # list of int
#         masked_ids = [text_ids[p] for p in masked_pos]

#         # pad
#         n_pad = 2*self.max_tokens - n_tokens
#         text_ids = text_ids + [self.pad_token_id] * n_pad
#         text_atts = [1] * n_tokens + [0] * n_pad

#         text_ids_masked = text_ids_masked + [self.pad_token_id] * n_pad
#         n_pad = self.max_masks - len(masked_ids)
#         masked_pos = masked_pos + [0] * n_pad
#         masked_ids = masked_ids + [self.PAD_mask] * n_pad

#         return text_ids, text_atts, text_ids_masked, masked_pos, masked_ids

#     def collate_fn(self, batch):
#         batch_tensors = []
#         for x in zip(*batch):
#             if x[0] is None:
#                 batch_tensors.append(None)
#             elif isinstance(x[0], torch.Tensor):
#                 batch_tensors.append(torch.stack(x))
#             else:
#                 batch_tensors.append(torch.tensor(x, dtype=torch.long))

#         return batch_tensors