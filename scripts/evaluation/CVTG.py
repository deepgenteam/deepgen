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
from einops import rearrange
import inflect
from peft import PeftModel

p = inflect.engine()


class CVTGDataset(Dataset):
    def __init__(self, data_root="/share/dnk/benchmark"):
 
        self.data = []
        # CVTG loop logic
        for area in range(2, 6):
            for benchmark in ("CVTG", "CVTG-Style"):
                json_path = os.path.join(data_root, benchmark, f"{area}.json")
                
                if not os.path.exists(json_path):
                    print(f"Warning: File not found {json_path}")
                    continue

                with open(json_path, 'r', encoding='utf-8') as file:
                    json_data = json.load(file)
                
                data_list = json_data.get("data_list", [])
                
                for item in data_list:
            
                    self.data.append({
                        "prompt": item.get("prompt"),
                        "index": item.get("index"),
                        "benchmark": benchmark,
                        "area": area,
                        "carrier_list": item.get("carrier_list"),
                        "sentence_list": item.get("sentence_list")
                    })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--checkpoint', default=None, type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--data', default='evaluation/CVTG-2K', type=str, help="Root path containing CVTG/ and CVTG-Style/ folders")
    parser.add_argument('--output', default='result_CVTG', type=str)
    parser.add_argument("--cfg_prompt", type=str, default='')
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    parser.add_argument('--num_steps', type=int, default=50) 
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    accelerator = Accelerator()
    
    # Distributed init check
    message = [f"Hello this is GPU {accelerator.process_index}"]
    messages = gather_object(message)
    accelerator.print(f"Number of gpus: {accelerator.num_processes}")
    accelerator.print(messages)

    config = Config.fromfile("configs/models/deepgen_scb.py")

    print(f'Device: {accelerator.device}', flush=True)

 
    dataset = CVTGDataset(data_root=args.data)
    
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
        accelerator.print(f'Checkpoint contains {len(state_dict)} keys:')
     
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        accelerator.print(f'Missing parameters ({len(missing)})')
        accelerator.print(f'Unexpected parameters ({len(unexpected)})')
    
    model = model.to(device=accelerator.device)
    model = model.to(model.dtype)
    model.eval()

    dataloader = accelerator.prepare(dataloader)

    print(f'Number of samples: {len(dataloader)}', flush=True)

    if accelerator.is_main_process:
        os.makedirs(args.output, exist_ok=True)

    generator = torch.Generator(device=model.device).manual_seed(args.seed)

 
    n_samples = 1 

    for batch_idx, data_samples in tqdm(enumerate(dataloader), disable=not accelerator.is_main_process):
        
   
        prompt = [data_sample['prompt'].strip() for data_sample in data_samples] * n_samples
        cfg_prompt = [args.cfg_prompt] * len(prompt)

    
        images = model.generate(prompt=prompt, cfg_prompt=cfg_prompt, pixel_values_src=None,
                                cfg_scale=args.cfg_scale, num_steps=args.num_steps,
                                progress_bar=False,
                                generator=generator, height=args.height, width=args.width)
        
        # Rearrange output
        # shape: (batch * n) c h w -> batch n h w c
        images = rearrange(images, '(n b) c h w -> b n h w c', n=n_samples)

        images = torch.clamp(
            127.5 * images + 128.0, 0, 255).to("cpu", dtype=torch.uint8).numpy()

     
        for image_batch, data_sample in zip(images, data_samples):
       
            benchmark = data_sample['benchmark']
            area = data_sample['area']
            index = data_sample['index']
            
            # {base_dir}/{benchmark}/{area}/
            save_dir = os.path.join(args.output, benchmark, str(area))
            os.makedirs(save_dir, exist_ok=True)

        
            final_image = Image.fromarray(image_batch[0])
            save_path = os.path.join(save_dir, f"{index}.png")
            final_image.save(save_path)
            
           