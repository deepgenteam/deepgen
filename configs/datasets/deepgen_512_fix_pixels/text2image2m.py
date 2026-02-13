from src.datasets.text2image.caption_datasets import CaptionDataset
from src.datasets.collate_functions import collate_func_gen_txt_dynamic
from mmengine.config import read_base
from mmengine.dataset import InfiniteSampler
from xtuner.dataset import ConcatDataset

with read_base():
    from .processors import image_size, image_process

t2i_2m = dict(type=CaptionDataset,
              image_size=image_size,
              image_process=image_process,
              cap_source='prompt',
              data_path='data/text-to-image-2M/data/data_512_2M.json',
              cap_folder='data/text-to-image-2M/raw/data_512_2M',
              image_folder='data/text-to-image-2M/raw/data_512_2M',
              ceph_folder=None,
              ceph_config=None,)

t2i_10k = dict(type=CaptionDataset,
               image_size=image_size,
               image_process=image_process,
               cap_source='prompt',
               data_path='data/text-to-image-2M/data/data_1024_10K.json',
               cap_folder='data/text-to-image-2M/raw/data_1024_10K',
               image_folder='data/text-to-image-2M/raw/data_1024_10K',
               ceph_folder=None,
               ceph_config=None)

dataset = dict(
    type=ConcatDataset,
    datasets=[t2i_2m, t2i_10k]
)

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    pin_memory=True,
    dataset=dataset,
    sampler=dict(type=InfiniteSampler, shuffle=True),
    collate_fn=dict(type=collate_func_gen_txt_dynamic)
)