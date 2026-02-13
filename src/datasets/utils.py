import copy
import random
from xtuner.dataset.utils import get_bos_eos_token_ids
from xtuner.utils import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX, IMAGE_TOKEN_INDEX
import json
import math

INPUT_IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX
OUTPUT_IMAGE_TOKEN_INDEX = -300


def resize_image_fix_pixels(x, image_size, unit_image_size=32):
    # import pdb; pdb.set_trace()
    w, h = x.size
    ratio = image_size / ((h * w) ** 0.5)

    target_h = math.ceil(h * ratio / unit_image_size) * unit_image_size
    target_w = math.ceil(w * ratio / unit_image_size) * unit_image_size

    x = x.resize(size=(target_w, target_h))

    return x


def resize_image_dynamic(x, image_size, unit_image_size=32):
    w, h = x.size
    if w >= h and w >= image_size:
        target_w = image_size
        target_h = h * (target_w / w)
        target_h = math.ceil(target_h / unit_image_size) * unit_image_size

    elif h >= w and h >= image_size:
        target_h = image_size
        target_w = w * (target_h / h)
        target_w = math.ceil(target_w / unit_image_size) * unit_image_size

    else:
        target_h = math.ceil(h / unit_image_size) * unit_image_size
        target_w = math.ceil(w / unit_image_size) * unit_image_size

    x = x.resize(size=(target_w, target_h))

    return x



def crop2square(pil_img):
    width, height = pil_img.width, pil_img.height

    if width > height:
        y0, y1 = 0, height
        x0 = random.randint(0, width - height)    # [0, w - h]
        x1 = x0 + height    # [h, w]
    else:
        x0, x1 = 0, width
        y0 = random.randint(0, height - width)   # [0, h - w]
        y1 = y0 + width     # [w, h]

    return pil_img.crop(box=(x0, y0, x1, y1))


def load_jsonl(json_file):
    with open(json_file) as f:
        lines = f.readlines()
    data = []
    for line in lines:
        data.append(json.loads(line))
    return data


def encode_fn(example,
              tokenizer,
              max_length=None,
              image_length=1,
              input_ids_with_output=True,
              with_image_token=False,
              truncation='right'):
    """We only support the following three scenarios:

    1. Incremental pretraining dataset.
        example['conversation'] = [
                {
                    'input': '',
                    'output': '### Human: Can you write xxx'
                }
            ]

    2. Single-turn conversation dataset.
        example['conversation'] = [
                {
                    'input': 'Give three tips for staying healthy.',
                    'output': '1.Eat a balanced diet xxx'
                }
            ]

    3. Multi-turn conversation dataset.
        example['conversation'] = [
                {
                    'input': 'Give three tips for staying healthy.',
                    'output': '1.Eat a balanced diet xxx'
                },
                {
                    'input': 'Please expand on the second point.',
                    'output': 'Here is an expanded explanation of the xxx'
                }
            ]
    """
    bos_token_id, eos_token_id = get_bos_eos_token_ids(tokenizer)
    is_multi_turn_conversation = len(example['conversation']) > 1
    if is_multi_turn_conversation:
        assert input_ids_with_output

    input_ids, labels = [], []
    next_needs_bos_token = True
    for single_turn_conversation in example['conversation']:
        input = single_turn_conversation['input']
        if DEFAULT_IMAGE_TOKEN in input and with_image_token:
            chunk_encode = [
                tokenizer.encode(chunk, add_special_tokens=False)
                for chunk in input.split(DEFAULT_IMAGE_TOKEN)
            ]
            assert len(chunk_encode) == 2
            input_encode = []
            for idx, cur_chunk_encode in enumerate(chunk_encode):
                input_encode.extend(cur_chunk_encode)
                if idx != len(chunk_encode) - 1:
                    # input_encode.append(IMAGE_TOKEN_INDEX)
                    input_encode += [IMAGE_TOKEN_INDEX] * image_length
        else:
            input_encode = tokenizer.encode(input, add_special_tokens=False)
        if next_needs_bos_token:
            input_ids += bos_token_id
            labels += [IGNORE_INDEX] * len(bos_token_id)
        input_ids += input_encode
        labels += [IGNORE_INDEX] * len(input_encode)
        if input_ids_with_output and 'output' in single_turn_conversation:
            # Add output
            output_with_loss = single_turn_conversation.get(
                'output_with_loss', True)
            output = single_turn_conversation['output']
            if DEFAULT_IMAGE_TOKEN in output and with_image_token:
                chunk_encode = [
                    tokenizer.encode(chunk, add_special_tokens=False)
                    for chunk in output.split(DEFAULT_IMAGE_TOKEN)
                ]
                assert len(chunk_encode) == 2
                output_encode = []
                for idx, cur_chunk_encode in enumerate(chunk_encode):
                    output_encode.extend(cur_chunk_encode)
                    if idx != len(chunk_encode) - 1:
                        output_encode += [IMAGE_TOKEN_INDEX] * image_length
            else:
                output_encode = tokenizer.encode(output, add_special_tokens=False)
            # output_encode = tokenizer.encode(output, add_special_tokens=False)
            input_ids += output_encode
            if output_with_loss:
                labels += copy.deepcopy(output_encode)
            else:
                labels += [IGNORE_INDEX] * len(output_encode)
            # Add EOS_TOKEN (with loss)
            if single_turn_conversation.get('need_eos_token', True):
                next_needs_bos_token = True
                input_ids += eos_token_id
                if output_with_loss:
                    labels += copy.deepcopy(eos_token_id)
                else:
                    labels += [IGNORE_INDEX] * len(eos_token_id)
            else:
                next_needs_bos_token = False
            # Add SEP (without loss)
            sep = single_turn_conversation.get('sep', '')
            if sep != '':
                sep_encode = tokenizer.encode(sep, add_special_tokens=False)
                input_ids += sep_encode
                labels += [IGNORE_INDEX] * len(sep_encode)

    if max_length is not None and len(input_ids) > max_length:
        if truncation == 'right':
            input_ids = input_ids[:max_length]
            labels = labels[:max_length]
        elif truncation == 'left':
            input_ids = input_ids[-max_length:]
            labels = labels[-max_length:]
        else:
            assert truncation is None
    return {'input_ids': input_ids, 'labels': labels}
