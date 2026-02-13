import os
import json
import torch
import argparse
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from xtuner.registry import BUILDER
from mmengine.config import Config
from xtuner.model.utils import guess_load_checkpoint
from accelerate.utils import gather_object
from einops import rearrange
import numpy as np
from pathlib import Path
from datasets import load_from_disk


class GEditBench(Dataset):
    def __init__(self, data_path, shard_id, total_shards, width=512, height=512):
        self.data_path = data_path
        self.width = width
        self.height = height
        self.shard_id = shard_id
        self.total_shards = total_shards
        
        # Load the GEdit-Bench dataset
        dataset = load_from_disk(self.data_path)
        idx_list = list(range(len(dataset)))
        self.idx_list = idx_list[self.shard_id::self.total_shards]
        
        self.data = dataset
        
    def __len__(self):
        return len(self.idx_list)
        
    
    def _process_image(self, image):
        image = image.resize(size=(self.height, self.width))
        pixel_values = torch.from_numpy(np.array(image)).float()
        pixel_values = pixel_values / 255
        pixel_values = 2 * pixel_values - 1
        pixel_values = rearrange(pixel_values, 'h w c -> c h w')
        return pixel_values

    def __getitem__(self, idx):
        data_dict = self.data[self.idx_list[idx]]
        
        key = data_dict['key']
        task_type = data_dict['task_type']
        instruction = data_dict['instruction']
        instruction_language = data_dict['instruction_language']
        image_file = data_dict['input_image']  # Assuming 'input_image' has the path to the image
        
        # Read and process the image
        image = image_file
        image_pixel_src = self._process_image(image)
        
        # Update the dictionary with additional information
        data_dict.update({
            'key': key,
            'instruction': instruction,
            'image_pixel_src': image_pixel_src,
            'idx': idx,
            "task_type":task_type,
            "instruction_language":instruction_language,
            "image_file":image_file,

        })
        
        return data_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--checkpoint', default=None, type=str)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--data', default='evaluation/GEdit-Bench', type=str)
    parser.add_argument('--output', default='result_gedit', type=str)
    parser.add_argument("--cfg_prompt", type=str, default="blurry, low quality, low resolution, distorted, deformed, broken content, missing parts, damaged details, artifacts, glitch, noise, pixelated, grainy, compression artifacts, bad composition, wrong proportion, incomplete editing, unfinished, unedited areas.")
    parser.add_argument("--cfg_scale", type=float, default=4.0)  
    parser.add_argument('--num_steps', type=int, default=50)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument('--shard_id', type=int, default=0)
    parser.add_argument('--total_shards', type=int, default=1)

    args = parser.parse_args()

    accelerator = Accelerator()
    message = [f"Hello this is GPU {accelerator.process_index}"]
    messages = gather_object(message)
    accelerator.print(f"Number of gpus: {accelerator.num_processes}")
    accelerator.print(messages)
    config = Config.fromfile("configs/models/deepgen_scb.py")
    print(f'Device: {accelerator.device}', flush=True)
    
    # Prepare dataset
    dataset = GEditBench(data_path=args.data, shard_id=args.shard_id, total_shards=args.total_shards, 
                         width=args.width, height=args.height)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            drop_last=False,
                            collate_fn=lambda x: x
                            )
    
    # Build the model
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
            prompts.append(data_sample['instruction'].strip())
            image_pixel_srcs.append(data_sample['image_pixel_src'][None])
        image_pixel_srcs = torch.stack(image_pixel_srcs).to(model.dtype)
        images = model.generate(prompt=prompts, cfg_prompt=[args.cfg_prompt] * len(prompts),
                                pixel_values_src=image_pixel_srcs,
                                cfg_scale=args.cfg_scale, num_steps=args.num_steps,
                                generator=generator, height=args.height, width=args.width,
                                progress_bar=False)

        images = rearrange(images, 'b c h w -> b h w c')

        images = torch.clamp(127.5 * images + 128.0, 0, 255
                             ).to('cpu', dtype=torch.uint8).numpy()

        for image, data_sample in zip(images, data_samples):
            del data_sample['image_pixel_src']
            key = data_sample.pop('key')
            task_type = data_sample.pop('task_type')
            image_file = data_sample.pop('image_file')
            instruction_language = data_sample.pop('instruction_language')
            outpath = f"{args.output}/fullset/{task_type}/{instruction_language}/{key}.png"
            save_path_fullset_source_image = f"{args.output}/fullset/{task_type}/{instruction_language}/{key}_SRCIMG.png"
            os.makedirs(os.path.dirname(outpath), exist_ok=True)
            os.makedirs(os.path.dirname(save_path_fullset_source_image), exist_ok=True)
            Image.fromarray(image).save(outpath)
            image_file.save(save_path_fullset_source_image)
