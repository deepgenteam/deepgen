from mmengine.config import read_base
from mmengine.dataset import InfiniteSampler
from src.datasets.collate_functions import collate_func_gen_txt_dynamic
from src.datasets.text2image.blip3_o import BLIP3oDataset
from xtuner.dataset import ConcatDataset

with read_base():
    from .processors import image_size, image_process

image_process = 'fix_pixels'
dataset_blip3o60k = dict(type=BLIP3oDataset,
               image_size=image_size,
               data_path='data/BLIP3o/blip3o_60k.json',
               image_folder ='data/BLIP3o',
               image_process = image_process,
               ceph_folder=None,
               ceph_config=None)

dataset_share4oimg = dict(type=BLIP3oDataset,
               image_size=image_size,
               data_path='data/ShareGPT-4o-Image/share_4o_img.json',
               image_folder = "data/ShareGPT-4o-Image/t2i" ,
               image_process =image_process,
               ceph_folder=None,
               ceph_config=None)

dataset_echo4oimg = dict(type=BLIP3oDataset,
               image_size=image_size,
               data_path='data/Echo4o/echo-4o-image_t2i.json',
               image_folder = "data/Echo4o" ,
               image_process =image_process,
               ceph_folder=None,
               ceph_config=None)

dataset_open4oimg = dict(type=BLIP3oDataset,
               image_size=image_size,
               data_path='data/OpenGPT-4o-Image/OpenGPT-4o-Image.json',
               image_folder = "data/OpenGPT-4o-Image/t2i" ,
               image_process= image_process,
               ceph_folder=None,
               ceph_config=None)







dataset_reason_img = dict(type=BLIP3oDataset,
               image_size=image_size,
               data_path='data/unireason/reson_t2i.json',
               image_folder = "data/unireason/t2i" ,
               image_process = image_process,
               ceph_folder=None,
               ceph_config=None)

dataset_banana = dict(type=BLIP3oDataset,
               image_size=image_size,
               data_path='data/banana/banana-50k.json',
               image_folder = "data/banana" ,
               image_process = image_process,
               ceph_folder=None,
               ceph_config=None)

dataset_text = dict(type=BLIP3oDataset,
               image_size=image_size,
               data_path='data/text_render/text.json',
               image_folder = "data/text_render" ,
               image_process = image_process,
               ceph_folder=None,
               ceph_config=None)



dataset_poster = dict(type=BLIP3oDataset,
               image_size=image_size,
               data_path='data/poster/data.json',
               image_folder = "data/poster" ,
               image_process = image_process,
               ceph_folder=None,
               ceph_config=None)



dataset = dict(
    type=ConcatDataset,
    datasets=[dataset_blip3o60k,dataset_share4oimg,dataset_echo4oimg,dataset_open4oimg,dataset_reason_img,dataset_banana,dataset_text,dataset_poster],
)

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    pin_memory=True,
    dataset=dataset,
    sampler=dict(type=InfiniteSampler, shuffle=True),
    collate_fn=dict(type=collate_func_gen_txt_dynamic)
)
