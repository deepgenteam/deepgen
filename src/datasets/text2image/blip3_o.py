import torch
import os
from src.datasets.text2image.caption_datasets import CaptionDataset
from PIL import Image

class BLIP3oDataset(CaptionDataset):
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
                if self.image_folder is not None:
                    image = Image.open(os.path.join(self.image_folder,data_sample['image_path'])).convert('RGB')
                else:
                    image = Image.open(data_sample['image_path']).convert('RGB')
                data = self._process_image(image)

                caption = data_sample['txt']

            # print(caption)
            data["pixel_init"] = image
            data.update(self._process_text(caption))
            data.update(image_dir=self.image_folder, image_file=None,
                        type='text2image',text=caption)

            return data

        except Exception as e:
            print(f"Error when reading {self.data_path}:{self.data_list[idx]}: {e}", flush=True)
            return self._retry()
