import torch
import numpy as np
from einops import rearrange
from src.datasets.utils import crop2square
from src.datasets.text2image.caption_datasets import CaptionDataset
from PIL import Image
import os

class ImageEditDataset(CaptionDataset):
    def _process_image(self, image):
        assert self.image_process != 'crop2square'
        return super()._process_image(image)['pixel_values']
        # image = image.resize(size=(self.image_size, self.image_size))
        # pixel_values = torch.from_numpy(np.array(image)).float()
        # pixel_values = pixel_values / 255
        # pixel_values = 2 * pixel_values - 1
        # pixel_values = rearrange(pixel_values, 'h w c -> c h w')
        # return pixel_values

    def _process_text(self, text):
        prompt_template = self.prompt_template
        image_tokens = prompt_template['IMG_START_TOKEN'] + \
                       prompt_template['IMG_CONTEXT_TOKEN'] * self.image_length + \
                       prompt_template['IMG_END_TOKEN']
        prompt = f'{image_tokens}\n{text}'
        prompt = self.prompt_template['INSTRUCTION'].format(input=prompt)
        if self.prompt_template.get('IMG_START_TOKEN_FOR_GENERATION', True):
            prompt += prompt_template['IMG_START_TOKEN']
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt', **self.tokenizer_kwargs)[0]

        return dict(input_ids=input_ids)

    def __getitem__(self, idx):
        if self.debug:
            idx = 0
        try:
            data_sample = self.data_list[idx]
            if self.image_folder is not None:
                source_image = Image.open(os.path.join(self.image_folder,data_sample['input_image'][0])).convert('RGB')
                target_image = Image.open(os.path.join(self.image_folder,data_sample['output_image'])).convert('RGB')
            else:
                source_image = Image.open(data_sample['input_image'][0]).convert('RGB')
                target_image = Image.open(data_sample['output_image']).convert('RGB')
            # prompt = self._read_json(data_sample['annotation'])[self.cap_source]
            prompt = data_sample['instruction']

            pixel_values_src = self._process_image(source_image)
            pixel_values = self._process_image(target_image)

            data = self._process_text(prompt) if self.tokenizer is not None else dict()

            data.update(
                pixel_values_src=pixel_values_src, pixel_values=pixel_values,
                image_dir=self.image_folder,type='image2image', text=prompt)

            return data

        except Exception as e:
            print(f"Error when reading {self.data_path}:{self.data_list[idx]}: {e}", flush=True)
            return self._retry()


class ReconstructDataset(CaptionDataset):
    def _process_image(self, image):
        assert self.image_process != 'crop2square'
        return super()._process_image(image)['pixel_values']

    def __getitem__(self, idx):
        if self.debug:
            idx = 0
        try:
            data_sample = self.data_list[idx]
            image = self._read_image(data_sample['image']).convert('RGB')
            prompt = "Keep the image as it is."
            pixel_values = pixel_values_src = self._process_image(image)

            data = self._process_text(prompt) if self.tokenizer is not None else dict()

            data.update(
                pixel_values_src=pixel_values_src, pixel_values=pixel_values,
                image_dir=self.image_folder, image_file=data_sample['image'],
                type='image2image', text=prompt)

            return data

        except Exception as e:
            print(f"Error when reading {self.data_path}:{self.data_list[idx]}: {e}", flush=True)
            return self._retry()
