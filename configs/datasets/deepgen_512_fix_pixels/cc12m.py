from mmengine.config import read_base
from mmengine.dataset import InfiniteSampler
from src.datasets.collate_functions import collate_func_gen_txt_dynamic
from src.datasets.text2image.caption_datasets import CaptionDataset


with read_base():
    from .processors import image_size, image_process

dataset = dict(type=CaptionDataset,
               image_size=image_size,
               image_process=image_process,
               cap_folder='data/cc12m/captions',
               data_path='data/cc12m/data.json',
               image_folder='data/cc12m/raw',
               ceph_folder=None,
               ceph_config=None)

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    pin_memory=True,
    dataset=dataset,
    sampler=dict(type=InfiniteSampler, shuffle=True),
    collate_fn=dict(type=collate_func_gen_txt_dynamic)
)
