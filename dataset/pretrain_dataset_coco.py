import os
import json
import random
from random import random as rand
from transformers import XLMRobertaTokenizer, AutoTokenizer
import torch
from PIL import Image
from torch.utils.data import Dataset
import re
from torchvision.transforms.functional import hflip
import copy 
from random import randint, shuffle


def build_tokenizer(text_encoder: str):
    if 'xlm' in text_encoder:
        tokenizer = XLMRobertaTokenizer.from_pretrained(text_encoder)
    else:
        tokenizer = AutoTokenizer.from_pretrained(text_encoder)
    tokenizer.add_special_tokens({'bos_token': tokenizer.cls_token, 'eos_token': tokenizer.sep_token})
    return tokenizer


def pre_caption(caption, max_words):
    caption_raw = caption
    caption = re.sub(
        r"([,.'!?\"()*#:;~])",
        ' ',
        caption.lower(),
    ).replace('-', ' ').replace('/', ' ').replace('<person>', 'person')

    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n')
    caption = caption.strip(' ')

    # truncate caption
    caption_words = caption.split(' ')
    if len(caption_words) > max_words:
        caption = ' '.join(caption_words[:max_words])

    if not len(caption):
        raise ValueError(f"pre_caption yields invalid text (raw: {caption_raw})")

    return caption

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
        special_pos = set([0]) 
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


class cap_dataset(Dataset):
    def __init__(self, ann_file, transform, vqa_root=None, split="train", max_words=30, answer_list='',
                 text_encoder=''):

        self.careful_hflip = True

        self.split = split
        self.images = {}
        self.ann = {}
        if isinstance(ann_file, str):
            ann_file = [ann_file]
        elif not isinstance(ann_file, list):
            raise ValueError

        
        for f in ann_file:
            data = json.load(open(f, 'r'))
            images = data['images']
            for image in images:
                self.images[image['id']] = image
                # self.ann.update({image['id'] : []})

            self.max_ques_words = 50

            for ann in data['annotations']:
                if ann['image_id'] in self.images.keys():
                    # if split == 'valid':
                    #     if ann['image_id'] in self.ann: 
                    #         self.ann[ann['image_id']].append(ann)
                    #     else:
                    #         self.ann[ann['image_id']] = [ann]
                    # elif split == 'train':  
                    self.ann[ann['id']] = ann 

        
        self.transform = transform
        self.vqa_root = vqa_root

        self.tokenizer = build_tokenizer(text_encoder)

        self.pad_token_id = self.tokenizer.pad_token_id
        self.eos_token = self.tokenizer.sep_token


        self.add_eos = True  # always add eos
        self.cls_token = self.tokenizer.cls_token
        self.eos_token = self.tokenizer.sep_token
        self.pad_token_id = self.tokenizer.pad_token_id
        self.mask_token_id = self.tokenizer.mask_token_id

        # self.mask_generator = TextMaskingGenerator(self.tokenizer, config['mask_prob'],
        #                                            config['max_masks'], config['skipgram_prb'],
        #                                            config['skipgram_size'], mask_whole_word=False)

        self.PAD_mask = -100  # loss will ignore this
        self.max_words = max_words
        self.max_tokens = max_words

        mask_prob = 0.4
        max_masks = 12
        skipgram_prb = 0.2
        skipgram_size = 3

        self.max_masks = max_masks
        self.transform = transform
        # self.image_res = config['image_res']
        # self.patch_size = config['patch_size']
        # self.num_patch = int(self.image_res / self.patch_size)



        self.mask_generator = TextMaskingGenerator(self.tokenizer, mask_prob,
                                    max_masks, skipgram_prb,
                                    skipgram_size, mask_whole_word=False)

        if split == 'valid':
            self.max_words = 50  # do not limit question length during test
        elif split == 'train':
            #self.ann = self.ann[:int(0.8 * len(self.ann))]
            # self.ann = [self.ann[i] for i in range(len(self.ann)) if i % 10 not in [8,9] ]
            pass
        elif split == 'test':
            self.max_words = 50

        
    def __len__(self):
        return len(self.ann.keys())

    def left_or_right_in(self, question, answer):
        def _func(s):
            if ('left' in s) or ('right' in s):
                return True
            else:
                return False

        if _func(question):
            return True

        if isinstance(answer, list):
            for ans in answer:
                if _func(ans):
                    return True
        else:
            if _func(answer):
                return True

        return False

    def __getitem__(self, index):
        k = list(self.ann.keys())[index]
        ann = self.ann[k]
        # if self.split in ['valid']:
        #     image_path = os.path.join(self.vqa_root, self.images[ann[0]['image_id']]['file_name'])
        # else:
        image_path = os.path.join(self.vqa_root, self.images[ann['image_id']]['file_name'])

        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        if self.split in ['test']:
            return image
        elif self.split in ['valid','train']:
            text_ids, text_atts, text_ids_masked, masked_pos, masked_ids = self.preprocess(ann['caption'])
            return image, text_ids, text_atts, text_ids_masked, masked_pos, masked_ids
        else:
            raise NotImplementedError

    def preprocess(self, text):
        text = pre_caption(text, self.max_words)
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



def cap_collate_fn(batch):
    batch_tensors = []
    for x in zip(*batch):
        if x[0] is None:
            batch_tensors.append(None)
        elif isinstance(x[0], torch.Tensor):
            batch_tensors.append(torch.stack(x))
        else:
            batch_tensors.append(torch.tensor(x, dtype=torch.long))

    return batch_tensors
