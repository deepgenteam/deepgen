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
from src.datasets.utils import load_jsonl
from einops import rearrange
import inflect
p = inflect.engine()


class CoREBench(Dataset):
    def __init__(self, data_path):
        self.data = load_jsonl(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_dict = copy.deepcopy(self.data[idx])

        text = data_dict['Prompt']
        ID = data_dict['ID']

        data_dict.update(idx=idx, text=text,ID=ID)

        return data_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    # parser.add_argument('config', required=True, help='config file path.')
    parser.add_argument('--checkpoint', default=None, type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--data', default='evaluation/T2I-CoReBench-main/corebench.jsonl', type=str)
    parser.add_argument('--output', default='result_corebench', type=str)
    parser.add_argument("--cfg_prompt", type=str, default='')
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    parser.add_argument('--num_steps', type=int, default=50) 
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
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

    dataset = CoREBench(data_path=args.data)
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

    for batch_idx, data_samples in tqdm(enumerate(dataloader), disable=not accelerator.is_main_process):
        device_idx = accelerator.process_index
       
        prompt = [data_sample['text'].strip() for data_sample in data_samples] * 4
        cfg_prompt = [args.cfg_prompt] * len(prompt)

        images = model.generate(prompt=prompt, cfg_prompt=cfg_prompt, pixel_values_src=None,
                                cfg_scale=args.cfg_scale, num_steps=args.num_steps,
                                progress_bar=False,
                                generator=generator, height=args.height, width=args.width)
        images = rearrange(images, '(n b) c h w -> b n h w c', n=4)

        images = torch.clamp(
            127.5 * images + 128.0, 0, 255).to("cpu", dtype=torch.uint8).numpy()

        # Save samples to disk as individual .png files
        for image, data_sample in zip(images, data_samples):
            sample_id = data_sample.pop('ID')
            os.makedirs(f"{args.output}/{'-'.join(sample_id.split('-')[:2])}", exist_ok=True)
            output_dir= f"{args.output}/{'-'.join(sample_id.split('-')[:2])}"
            sample_count =0
            for i, sub_image in enumerate(image):
                Image.fromarray(sub_image).save(os.path.join(output_dir, f"{sample_id}-{sample_count}.png"))
                sample_count += 1
