import torch
from xtuner.utils import DEFAULT_PAD_TOKEN_INDEX, IGNORE_INDEX
from typing import Dict, Sequence
from torch.nn.utils.rnn import pad_sequence
from functools import partial
from dataclasses import dataclass


def collate_func_img2img(instances: Sequence[Dict],
                         pad_index: int = DEFAULT_PAD_TOKEN_INDEX):
    pixel_values_src_list, pixel_values_list, input_ids_list, texts = [], [], [], []
    for instance in instances:
        pixel_values_src_ = instance.pop('pixel_values_src')
        if isinstance(pixel_values_src_, torch.Tensor):
            pixel_values_src_ = [pixel_values_src_]
        pixel_values_src_list += pixel_values_src_
        pixel_values_list.append(instance.pop('pixel_values'))
        input_ids_list.append(instance.pop('input_ids'))
        texts.append(instance.pop('text', None))

    ori_length = [len(ids) for ids in input_ids_list]
    pad_length = max(ori_length)
    attention_mask = torch.zeros(len(instances), pad_length, dtype=torch.bool)
    input_ids = torch.full(size=(len(instances), pad_length),
                           fill_value=pad_index, dtype=torch.long)

    # left padding for editing
    for i, length in enumerate(ori_length):
        attention_mask[i, -length:] = True
        input_ids_i = input_ids_list[i]
        if not isinstance(input_ids_i, torch.Tensor):
            input_ids_i = torch.tensor(input_ids_i, dtype=torch.long)
        input_ids[i, -length:] = input_ids_i

    pixel_values = torch.stack(pixel_values_list)
    pixel_values_src = torch.stack(pixel_values_src_list)

    data_dict = dict(input_ids=input_ids, attention_mask=attention_mask,
                     pixel_values=pixel_values, pixel_values_src=pixel_values_src, texts=texts)

    return {'data': data_dict, 'data_samples': None}


def collate_func_img2img_text(instances: Sequence[Dict]):
    pixel_values_src_list, pixel_values_list, texts = [], [], []
    for instance in instances:
        pixel_values_src_ = instance.pop('pixel_values_src')
        if isinstance(pixel_values_src_, torch.Tensor):
            pixel_values_src_ = [pixel_values_src_]
        pixel_values_src_list += pixel_values_src_
        pixel_values_list.append(instance.pop('pixel_values'))
        texts.append(instance.pop('text'))

    pixel_values = torch.stack(pixel_values_list)
    pixel_values_src = torch.stack(pixel_values_src_list)

    data_dict = dict(pixel_values=pixel_values, pixel_values_src=pixel_values_src, texts=texts)

    return {'data': data_dict, 'data_samples': None}


def collate_func_img2img_txt_dynamic(instances: Sequence[Dict]):
    pixel_values_src, pixel_values, texts = [], [], []
    for instance in instances:
        pixel_values_src_ = instance.pop('pixel_values_src')
        if isinstance(pixel_values_src_, torch.Tensor):    # only has one ref image
            pixel_values_src_ = [pixel_values_src_]
        pixel_values_src.append(pixel_values_src_)
        pixel_values.append(instance.pop('pixel_values'))
        texts.append(instance.pop('text'))

    data_dict = dict(pixel_values=pixel_values, pixel_values_src=pixel_values_src, texts=texts)

    return {'data': data_dict, 'data_samples': None}


def collate_func_gen_txt_dynamic(instances: Sequence[Dict]):
    pixel_values, texts = [], []
    for example in instances:
        pixel_values.append(example.pop('pixel_values'))
        texts.append(example.pop('text'))

    data_dict = dict(pixel_values=pixel_values, texts=texts)

    return {'data': data_dict, 'data_samples': None}



def collate_func_gen(instances: Sequence[Dict],
                     pad_index: int = DEFAULT_PAD_TOKEN_INDEX):
    pixel_values, input_ids, input_lengths, texts, pixel_init = [], [], [], [], []
    for example in instances:
        pixel_values.append(example.pop('pixel_values'))
        input_lengths.append(len(example['input_ids']))
        input_ids.append(example.pop('input_ids'))
        texts.append(example.pop('text', None))
        pixel_init.append(example.pop('pixel_init'))

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_index)
    attention_mask = torch.zeros_like(input_ids).bool()
    for i in range(len(input_ids)):
        attention_mask[i, :input_lengths[i]] = True

    data_dict = dict(pixel_values=torch.stack(pixel_values),
                     pixel_init = pixel_init,
                     input_ids=input_ids,
                     attention_mask=attention_mask,
                     texts=texts)

    return {'data': data_dict, 'data_samples': None}


def collate_func_gen_text(instances: Sequence[Dict]):
    pixel_values, texts = [], []
    for example in instances:
        pixel_values.append(example.pop('pixel_values'))
        texts.append(example.pop('text'))

    data_dict = dict(pixel_values=torch.stack(pixel_values), texts=texts)

    return {'data': data_dict, 'data_samples': None}


def collate_func_gen_tokens(instances: Sequence[Dict],
                            pad_index: int = DEFAULT_PAD_TOKEN_INDEX):
    image_tokens, input_ids, input_lengths, texts = [], [], [], []
    for example in instances:
        image_tokens.append(example.pop('image_tokens'))
        input_lengths.append(len(example['input_ids']))
        input_ids.append(example.pop('input_ids'))
        texts.append(example.pop('text', None))

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_index)
    attention_mask = torch.zeros_like(input_ids).bool()
    for i in range(len(input_ids)):
        attention_mask[i, :input_lengths[i]] = True

    data_dict = dict(image_tokens=torch.stack(image_tokens),
                     input_ids=input_ids,
                     attention_mask=attention_mask,
                     texts=texts)

    return {'data': data_dict, 'data_samples': None}


def collate_func_gen_latents(instances: Sequence[Dict],
                             pad_index: int = DEFAULT_PAD_TOKEN_INDEX):
    image_latents, input_ids, input_lengths, texts = [], [], [], []
    for example in instances:
        image_latents.append(example.pop('image_latents'))
        input_lengths.append(len(example['input_ids']))
        input_ids.append(example.pop('input_ids'))
        texts.append(example.pop('text', None))

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_index)
    attention_mask = torch.zeros_like(input_ids).bool()
    for i in range(len(input_ids)):
        attention_mask[i, :input_lengths[i]] = True

    data_dict = dict(image_latents=torch.stack(image_latents),
                     input_ids=input_ids,
                     attention_mask=attention_mask,
                     texts=texts)

    return {'data': data_dict, 'data_samples': None}


def collate_func_gen_text_latents(instances: Sequence[Dict]):
    image_latents, texts = [], []
    for example in instances:
        image_latents.append(example.pop('image_latents'))
        texts.append(example.pop('text', None))

    data_dict = dict(image_latents=torch.stack(image_latents), texts=texts)

    return {'data': data_dict, 'data_samples': None}


def collate_func_und(instances, pad_index=DEFAULT_PAD_TOKEN_INDEX):
    input_ids_list, labels_list, pixel_values_list = [], [], []

    for sample in instances:
        input_ids_list.append(torch.LongTensor(sample['input_ids']))
        labels_list.append(torch.LongTensor(sample['labels']))

        if 'pixel_values' in sample:
            pixel_values_list.append(sample['pixel_values'])

    ori_length = [len(input_ids_) for input_ids_ in input_ids_list]
    # right padding
    if len(instances) > 1:
        input_ids = pad_sequence(
            input_ids_list, batch_first=True, padding_value=pad_index)
        labels = pad_sequence(
            labels_list, batch_first=True, padding_value=IGNORE_INDEX)
    else:
        input_ids = torch.stack(input_ids_list)
        labels = torch.stack(labels_list)

    attention_mask = torch.zeros_like(input_ids).bool()
    for i, length in enumerate(ori_length):
        attention_mask[i, :length] = True        # right padding

    data_dict = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'pixel_values': torch.stack(pixel_values_list) if len(pixel_values_list) > 0 else None
    }

    return {'data': data_dict, 'data_samples': None}


class CollateConcat(object):
    def __init__(self, collate_fns, keys):
        self.keys = keys
        self.collate_fns = {}
        for key, collate_fn in zip(keys, collate_fns):
            func = collate_fn.pop('type')
            self.collate_fns[key] = partial(func, **collate_fn)

    def __call__(self, data_samples):
        data_samples = [data_sample for data_sample in data_samples if len(data_sample) > 0]
        data_dict = {}
        key = data_samples[0]['type']
        data_dict[key] = self.collate_fns[key](data_samples)['data']

        return {'data': data_dict, 'data_samples': None}
