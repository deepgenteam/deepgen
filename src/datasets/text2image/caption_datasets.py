from torch.utils.data import Dataset
from PIL import Image
import os
import io
import json
import random
import torch
try:
    from aoss_client.client import Client
except:
    try:
        from petrel_client.client import Client
    except:
        Client = None
from glob import glob
from xtuner.registry import BUILDER
from src.datasets.utils import crop2square, resize_image_fix_pixels, resize_image_dynamic
from einops import rearrange
import numpy as np


class CaptionDataset(Dataset):
    def __init__(self,
                 data_path,
                 image_folder=None,
                 debug=False,
                 image_processor=None,
                 image_process='crop2square',
                 ceph_folder=None,
                 latents_ceph_folder=None,
                 ceph_config=None,
                 tokenizer=None,
                 prompt_template=None,
                 max_length=2048,
                 min_image_size=80,
                 image_size=256,
                 image_length=256,
                 unit_image_size=32,
                 image_tokens_folder=None,
                 image_latents_folder=None,
                 cap_folder=None,
                 cap_source='caption',
                 tokenizer_kwargs=dict(add_special_tokens=True),
                 unconditional=0.1
                 ):
        super().__init__()
        self.data_path = data_path
        self._load_data(data_path)
        self.image_folder = image_folder
        self.cap_folder = cap_folder
        self.cap_source = cap_source
        self.debug = debug

        if image_processor is not None:
            self.image_processor = BUILDER.build(image_processor)
        else:
            self.image_processor = None

        if tokenizer is not None:
            self.tokenizer = BUILDER.build(tokenizer)
        else:
            self.tokenizer = None
        self.prompt_template = prompt_template

        self.max_length = max_length
        self.image_process = image_process
        self.image_length = image_length
        self.image_tokens_folder = image_tokens_folder
        self.image_latents_folder = image_latents_folder
        self.min_image_size = min_image_size
        self.image_size = image_size
        self.unit_image_size = unit_image_size
        self.unconditional = unconditional
        self.tokenizer_kwargs = tokenizer_kwargs

        self.FILE_CLIENT = None
        self.ceph_folder = ceph_folder
        self.ceph_config = ceph_config
        self.latents_ceph_folder = latents_ceph_folder
        self.use_ceph = ((Client is not None) and (ceph_config is not None) and os.path.exists(ceph_config))

    def _load_data(self, data_path: str):      # image path and annotation path are saved in a json file
        if data_path.endswith('.json'):
            with open(data_path, 'r') as f:
                self.data_list = json.load(f)
        else:
            json_files = glob(f"{data_path}/*.json")
            data_list = []
            for json_file in json_files:
                with open(json_file, 'r') as f:
                    data_list += json.load(f)

            self.data_list = data_list

        print(f"Load {len(self.data_list)} data samples from {data_path}", flush=True)

    def __len__(self):
        return len(self.data_list)

    def _read_ceph(self, ceph_path):
        if self.FILE_CLIENT is None:
            self.FILE_CLIENT = Client(self.ceph_config)
        data_bytes = self.FILE_CLIENT.get(ceph_path)

        return io.BytesIO(data_bytes)

    def _read_image(self, image_file):
        if self.image_folder is None:
            assert self.use_ceph
            assert self.ceph_folder is not None
            image = Image.open(
                self._read_ceph(
                    os.path.join(self.ceph_folder, image_file)
                )
            )
        else:
            image = Image.open(
                os.path.join(self.image_folder, image_file)
            )
        assert image.width > self.min_image_size and image.height > self.min_image_size, f"Image: {image.size}"
        assert image.width / image.height > 0.1, f"Image: {image.size}"
        assert image.width / image.height < 10, f"Image: {image.size}"
        return image.convert('RGB')

    def _read_json(self, annotation_file):
        if self.cap_folder is None:
            assert self.use_ceph
            assert self.ceph_folder is not None
            annotation = json.load(
                self._read_ceph(
                    os.path.join(self.ceph_folder, annotation_file)
                )
            )
        else:
            with open(os.path.join(self.cap_folder, annotation_file), 'r') as f:
                annotation = json.load(f)

        return annotation

    def _process_image(self, image):
        data = dict()
        if self.image_process == 'crop2square':
            image = crop2square(image)
            image = image.resize(size=(self.image_size, self.image_size))
        elif self.image_process == 'dynamic':   # dynamic and make sure the largest edge <= self.image_size
            image = resize_image_dynamic(x=image, image_size=self.image_size, unit_image_size=self.unit_image_size)
        elif self.image_process == 'fix_pixels': # fix pixels contain radio of image
            # import pdb; pdb.set_trace()
            image = resize_image_fix_pixels(x=image, image_size=self.image_size, unit_image_size=self.unit_image_size)
        elif self.image_process == 'resize2square':
            image = image.resize(size=(self.image_size, self.image_size))
        else:
            raise NotImplementedError

        # assert image.width <= self.image_size
        # assert image.height <= self.image_size
        assert image.width % self.unit_image_size == 0
        assert image.height % self.unit_image_size == 0

        pixel_values = torch.from_numpy(np.array(image)).float()
        pixel_values = pixel_values / 255
        pixel_values = 2 * pixel_values - 1
        pixel_values = rearrange(pixel_values, 'h w c -> c h w')

        data.update(pixel_values=pixel_values)
        return data

    def _process_text(self, text):
        if self.tokenizer is None:
            return {}
        if random.uniform(0, 1) < self.unconditional:
            prompt = self.prompt_template['CFG']
        else:
            prompt = self.prompt_template['GENERATION'].format(input=text.strip())

        prompt = self.prompt_template['INSTRUCTION'].format(input=prompt)
        if self.prompt_template.get('IMG_START_TOKEN_FOR_GENERATION', True):
            prompt += self.prompt_template['IMG_START_TOKEN']
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt', **self.tokenizer_kwargs)[0]

        return dict(input_ids=input_ids[:self.max_length])

    def _retry(self):
        return self.__getitem__(random.choice(range(self.__len__())))

    def __getitem__(self, idx):
        if self.debug:
            idx = 0
        try:
            data_sample = self.data_list[idx]

            if self.image_tokens_folder is not None:
                image_tokens = torch.load(os.path.join(self.image_tokens_folder,
                                                       data_sample['image'] + '.pt')).long()
                data = dict(image_tokens=image_tokens)
            elif self.latents_ceph_folder is not None:
                image_latents = torch.load(
                    self._read_ceph(
                        os.path.join(
                            self.latents_ceph_folder, data_sample['image'] + '.pt'
                        )
                    )
                )
                data = dict(image_latents=image_latents)
            elif self.image_latents_folder is not None:
                image_latents = torch.load(os.path.join(self.image_latents_folder,
                                                        data_sample['image'] + '.pt'))
                data = dict(image_latents=image_latents)
            else:
                image = self._read_image(data_sample['image']).convert('RGB')
                data = self._process_image(image)

            caption = self._read_json(data_sample['annotation'])[self.cap_source]
            # caption = self._read_json(data_sample['annotation'])
            # print(caption)

            data.update(self._process_text(caption))
            data['pixel_init'] = image
            data.update(image_dir=self.image_folder, image_file=data_sample['image'],
                        type='text2image',text=caption)

            return data

        except Exception as e:
            print(f"Error when reading {self.data_path}:{self.data_list[idx]}: {e}", flush=True)
            return self._retry()
