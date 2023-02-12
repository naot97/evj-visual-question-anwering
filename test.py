# from transformers import BertLMHeadModel, BertTokenizer, BertConfig
# import torch

# tokenizer = BertTokenizer.from_pretrained("distilbert-base-cased")
# model = BertLMHeadModel.from_pretrained("distilbert-base-cased")
# checkpoint = torch.load('data/distilbert-base-25lang-cased/pytorch_model.bin', map_location='cpu')


# prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
# choice0 = "It is eaten with a fork and a knife."
# choice1 = "It is eaten while held in the hand."
# labels = torch.tensor(0).unsqueeze(0)  # choice0 is correct (according to Wikipedia ;)), batch size 1

# encoding = tokenizer([[prompt, choice0], [prompt, choice1]], return_tensors="pt", padding=True)
# for k, v in encoding.items():
#     print(k,v.shape)
# outputs = model(**{k: v.unsqueeze(0) for k, v in encoding.items()}, labels=labels)  # batch size is 1

# # the linear classifier still needs to be trained
# loss = outputs.loss
# logits = outputs.logits

# print(logits)

import json

data = json.load(open('pre_train/STAIR-captions/stair_captions_v1.2_train.json', 'r'))
# data = json.load(open('pre_train/UIT-ViIC/uitviic_captions_train2017.json', 'r'))
# data = json.load(open('pre_train/annotations/captions_train2017.json', 'r'))

print(data.keys())
print(data['images'][0])
print(data['annotations'][0])
