import json
import os
import copy
import torch
import argparse
from tqdm import tqdm
from xtuner.registry import BUILDER
from mmengine.config import Config
from xtuner.model.utils import guess_load_checkpoint
from accelerate import Accelerator
from accelerate.utils import gather_object
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
from einops import rearrange
import numpy as np
from pathlib import Path



class UniREditBench(Dataset):
    def __init__(self, data_path, width=512, height=512):
        self.data_path = data_path
        self.width = width
        self.height = height
        with open ('UniREditBench/data.json', "r") as f:
            self.data = json.load(f)
        
    def __len__(self):
        return len(self.data)
        
    def _read_image(self, image_file):
        image = Image.open(
            os.path.join(self.data_path, image_file)
        )
        return image.convert('RGB')
    
    def _process_image(self, image):
        image = image.resize(size=(self.height, self.width))
        pixel_values = torch.from_numpy(np.array(image)).float()
        pixel_values = pixel_values / 255
        pixel_values = 2 * pixel_values - 1
        pixel_values = rearrange(pixel_values, 'h w c -> c h w')
        return pixel_values

    def __getitem__(self, idx):
        data_dict = copy.deepcopy(self.data[idx])
        image = self._read_image(data_dict['original_image_path'])
        image_pixel_src = self._process_image(image)
        data_dict.update(idx=idx, image_pixel_src=image_pixel_src, filename = f"{data_dict['idx']}.png" ,subdir=data_dict['name'])
        return data_dict
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    # parser.add_argument('config', help='config file path.')
    parser.add_argument('--checkpoint', default=None, type=str)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--data', default='evaluation/UniREditBench/data.json', type=str)
    parser.add_argument('--output', default='result_unireditbench', type=str)
    parser.add_argument("--cfg_prompt", type=str, default="blurry, low quality, low resolution, distorted, deformed, broken content, missing parts, damaged details, artifacts, glitch, noise, pixelated, grainy, compression artifacts, bad composition, wrong proportion, incomplete editing, unfinished, unedited areas.")
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    parser.add_argument('--num_steps', type=int, default=50)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    accelerator = Accelerator()
    # each GPU creates a string
    message = [f"Hello this is GPU {accelerator.process_index}"]
    # collect the messages from all GPUs
    messages = gather_object(message)
    # output the messages only on the main process with accelerator.print()
    accelerator.print(f"Number of gpus: {accelerator.num_processes}")
    accelerator.print(messages)
    config = Config.fromfile("configs/models/deepgen_scb.py")
    print(f'Device: {accelerator.device}', flush=True)
    
    dataset = UniREditBench(data_path=args.data, width=args.width, height=args.height)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            drop_last=False,
                            collate_fn=lambda x: x
                            )
    
    model = BUILDER.build(config.model)
    if args.checkpoint is not None:
        if args.checkpoint.endswith('.pt'):
            state_dict = torch.load(args.checkpoint)
        else:
            state_dict = guess_load_checkpoint(args.checkpoint)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        accelerator.print(f"Unexpected parameters: {unexpected}")
    model = model.to(device=accelerator.device)
    model = model.to(model.dtype)
    model.eval()

    dataloader = accelerator.prepare(dataloader)

    print(f'Number of samples: {len(dataloader)}', flush=True)
    
    if accelerator.is_main_process:
        os.makedirs(args.output, exist_ok=True)

    generator = torch.Generator(device=model.device).manual_seed(args.seed)

    for batch_idx, data_samples in tqdm(enumerate(dataloader),
                                        disable=not accelerator.is_main_process):

        prompts, image_pixel_srcs = [], []
        for data_sample in data_samples:
            if "rules" in data_sample.keys():
                prompt = data_sample["rules"]+" "+data_sample['instruction'].strip()
                prompts.append(prompt)
            else:
                prompts.append(data_sample['instruction'].strip())
            image_pixel_srcs.append(data_sample['image_pixel_src'][None])
        image_pixel_srcs = torch.stack(image_pixel_srcs).to(model.dtype)
        images = model.generate(prompt=prompts, cfg_prompt=[args.cfg_prompt] * len(prompts),
                                pixel_values_src=image_pixel_srcs,
                                cfg_scale=args.cfg_scale, num_steps=args.num_steps,
                                generator=generator, height=args.height, width=args.width,
                                progress_bar=False)

        # images = torch.cat([image_pixel_srcs, images], dim=-1)
        images = rearrange(images, 'b c h w -> b h w c')

        images = torch.clamp(127.5 * images + 128.0, 0, 255
                             ).to('cpu', dtype=torch.uint8).numpy()

        for image, data_sample in zip(images, data_samples):
            del data_sample['image_pixel_src']
            filename = data_sample.pop('filename')
            # os.makedirs(f"{args.output}/{sample_id:05d}", exist_ok=True)
            os.makedirs(args.output,exist_ok=True)
            subdir = data_sample.pop('subdir')
            
            save_dir = os.path.join(args.output, subdir)
            os.makedirs(save_dir, exist_ok=True)
            outpath = os.path.join(save_dir, filename)
            Image.fromarray(image).save(outpath)
