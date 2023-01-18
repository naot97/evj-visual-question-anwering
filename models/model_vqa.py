import os
import copy
import numpy as np
import torch
import torch.nn.functional as F

from models.xbert import BertConfig, BertLMHeadModel
from transformers import AutoModelForMaskedLM, RobertaForCausalLM
from models import XVLMBase, load_pretrained


def tile(x, dim, n_tile):
    init_dim = x.size(dim)
    repeat_idx = [1] * x.dim()
    repeat_idx[dim] = n_tile
    x = x.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(x, dim, order_index.to(x.device))


class VQAModel(XVLMBase):
    def __init__(self, config, tokenizer):
        super().__init__(config, load_vision_params=False, load_text_params=False,
                         use_contrastive_loss=False, use_matching_loss=False, use_mlm_loss=False, use_bbox_loss=False,
                         config_text=None)

        self.pad_token_id = tokenizer.pad_token_id
        config_enc = self.text_encoder.config
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.tokenizer = tokenizer

        self.num_text_layers = config_enc.fusion_layer
        self.num_cross_layers = config_enc.num_hidden_layers - config_enc.fusion_layer
        assert config['num_dec_layers'] == self.num_cross_layers, "initialization not implemented"

        config_dec = copy.deepcopy(config_enc)
        config_dec.encoder_width = config_enc.hidden_size
        config_dec.fusion_layer = 0  # start index
        config_dec.num_hidden_layers = config['num_dec_layers']
        self.cross_encoder_width = config_enc.encoder_width  # i.e. vision_width
        self.dec_encoder_width = config_enc.hidden_size
        self.prompt = ""

        self.text_decoder =  BertLMHeadModel(config=config_dec)
        if self.dec_encoder_width != self.cross_encoder_width:
            self.init_params = ['text_decoder.' + n for n, _ in self.text_decoder.named_parameters()
                                if ('crossattention.self.key' in n) or ('crossattention.self.value' in n)]
        else:
            self.init_params = []

    def load_pretrained(self, ckpt_rpath, config, is_eval=False):
        if is_eval:
            state_dict = load_pretrained(ckpt_rpath, config, is_eval=True)
        else:
            state_dict = load_pretrained(ckpt_rpath, config, load_text=False)

            print("### Loading pretrained text encoder", flush=True)
            for key in list(state_dict.keys()):
                if 'bert.' in key:
                    encoder_key = key.replace('bert.', '')
                    state_dict[encoder_key] = copy.deepcopy(state_dict[key])

                # intialize text decoder as multimodal encoder (last 6 layers of model.text_encoder)
                if 'text_encoder.' in key:
                    if 'layer.' in key:
                        encoder_keys = key.split('.')
                        layer_num = int(encoder_keys[4])
                        if layer_num < self.num_text_layers:
                            pass
                            #del state_dict[key]
                            #continue

                        elif (self.dec_encoder_width != self.cross_encoder_width) and \
                                (('crossattention.self.key' in key) or ('crossattention.self.value' in key)):
                            del state_dict[key]
                            continue
                        else:
                            decoder_layer_num = (layer_num - self.num_text_layers)
                            encoder_keys[4] = str(decoder_layer_num)
                            encoder_key = '.'.join(encoder_keys)
                    else:
                        encoder_key = key

                    decoder_key = encoder_key.replace('text_encoder', 'text_decoder')
                    state_dict[decoder_key] = copy.deepcopy(state_dict[key])
                    # del state_dict[key]

        msg = self.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % ckpt_rpath)
        # print("missing_keys: ", [p for p in msg.missing_keys if 'vision_encoder' not in p])
        print("unexpected_keys: ", [p for p in msg.unexpected_keys if 'text_encoder.'in p])

    def forward(self, image, quesiton, answer=None, k=None, weights=None, train=True):
        def _get_answers(out_ids):
            answers = []
            for output in out_ids:
                answer = self.tokenizer.decode(output, skip_special_tokens=True)
                answers.append(answer)
            return answers

        image_embeds = self.vision_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        question_output = self.text_encoder(quesiton.input_ids,
                                            attention_mask=quesiton.attention_mask,
                                            encoder_hidden_states=image_embeds,
                                            encoder_attention_mask=image_atts,
                                            output_attentions = True,
                                            return_dict=True)

        question_states = []
        question_atts = []
        for b in range(image.size(0)):
            question_states += [question_output.last_hidden_state[b]] * 1
            question_atts += [quesiton.attention_mask[b]] * 1
        question_states = torch.stack(question_states, 0)
        question_atts = torch.stack(question_atts, 0)
        if train:
            '''
            k: number of answers for each question
            weights: weight for each answer
            '''
            answer_targets = answer.input_ids.masked_fill(answer.input_ids == self.pad_token_id, -100)
            answer_output = self.text_decoder(answer.input_ids,
                                              attention_mask=answer.attention_mask,
                                              encoder_hidden_states=question_states,
                                              encoder_attention_mask=question_atts,
                                              labels=answer_targets,
                                              return_dict=True,
                                              )
            loss = answer_output.loss
            loss = loss.sum()
            return loss

        else:
            # inputs = self.tokenizer([self.prompt]*image.size(0),  return_tensors="pt").to(image.device)
            start_ids = torch.tensor( [[self.bos_token_id]] * image.size(0)).to(image.device)          
            tokens = self.text_decoder.generate(
                                    input_ids=start_ids,
                                    # attention_mask=inputs.attention_mask,
                                    encoder_hidden_states=question_states,
                                    encoder_attention_mask=question_atts,
                                    return_dict=True,
                                    eos_token_id=self.eos_token_id,
                                    pad_token_id =self.pad_token_id,
                                    min_length=3,
                                    repetition_penalty = 2.0,
                                    reduction='none',
                                    do_sample=True)
            return _get_answers(tokens)

