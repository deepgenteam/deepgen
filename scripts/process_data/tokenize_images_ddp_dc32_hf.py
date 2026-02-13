import torch.nn as nn
import os
import torch
import argparse
from tqdm import tqdm
from xtuner.registry import BUILDER
from mmengine.config import Config
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from diffusers import AutoencoderDC


def collate_func(instances):
    pixel_values = [example.pop('pixel_values') for example in instances]
    pixel_values = torch.stack(pixel_values)
    return pixel_values, instances


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoencoderDC.from_pretrained(
            'mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers').eval()

    @property
    def device(self):
        return self.model.device

    def forward(self, x):
        scaling_factor = self.model.config.scaling_factor if self.model.config.scaling_factor else 0.41407
        z = self.model.encode(x)[0] * scaling_factor

        return z


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('config', help='config file path.')
    # parser.add_argument('--output', default='', type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    args = parser.parse_args()

    # Initialization
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    num_replicas = dist.get_world_size()
    rank = dist.get_rank()
    print(f"ddp rank: {rank}, world_size: {num_replicas}")

    config = Config.fromfile(args.config)
    dataset = BUILDER.build(config.dataset)

    dataloader = DataLoader(dataset=dataset,
                            batch_size=args.batch_size,
                            sampler=DistributedSampler(dataset=dataset, shuffle=False,),
                            shuffle=False,
                            drop_last=False,
                            collate_fn=collate_func
                            )

    model = MyModel().eval()

    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    model = model.to(device).bfloat16()

    # # Model
    # model = DDP(model, device_ids=[device])
    # module = model.module

    print(f"Model on device: {model.device}", flush=True)

    local_rank = int(os.environ["LOCAL_RANK"])

    print(f"Local rank: {local_rank}", flush=True)

    for pixel_values_all, data_samples in tqdm(dataloader, disable=local_rank != 0):
        assert len(pixel_values_all) == len(data_samples)
        # assert len(pixel_values_all) == args.batch_size

        with torch.no_grad():
            image_latents_all = model(pixel_values_all.to(model.device).bfloat16())

        for image_latents, data_sample in zip(image_latents_all, data_samples):
            image_file = data_sample['image_file']
            image_dir = data_sample['image_dir']
            folder = os.path.dirname(image_file)
            save_dir = image_dir + f'_dc32_{config.image_size}'
            os.makedirs(os.path.join(save_dir, folder), exist_ok=True)
            save_path = os.path.join(save_dir, image_file + '.pt')

            if os.path.exists(save_path):
                continue

            # import pdb; pdb.set_trace()

            image_latents = image_latents.cpu()

            torch.save(image_latents, save_path)
