from src.datasets.collate_functions import collate_func_img2img_txt_dynamic
from mmengine.config import read_base
from mmengine.dataset import InfiniteSampler
from xtuner.dataset import ConcatDataset
from src.datasets.image2image.edit_datasets import ImageEditDataset, ReconstructDataset
from PIL import Image
with read_base():
    from .processors import image_size, image_process








Open4oEdit  = dict(type=ImageEditDataset,
               image_size=image_size,
               image_process=image_process,
               data_path='data/OpenGPT-4o-Image/editing.json',
               image_folder = "data/OpenGPT-4o-Image/editing" ,
               ceph_folder=None,
               ceph_config=None)

Share4oEdit  = dict(type=ImageEditDataset,
               image_size=image_size,
               image_process=image_process,
               data_path='data/ShareGPT-4o-Image/text_and_image_to_image.json',
               image_folder = "data/ShareGPT-4o-Image/editing" ,
               ceph_folder=None,
               ceph_config=None)

nano150Edit  = dict(type=ImageEditDataset,
               image_size=image_size,
               image_process=image_process,
               data_path='data/Nano-150k/data.json',
               image_folder = "data/Nano-150k" ,
               ceph_folder=None,
               ceph_config=None)

picobananaEdit  = dict(type=ImageEditDataset,
               image_size=image_size,
               image_process=image_process,
               data_path='data/pico-banana/sft_with_local_source_image_path.json',
               image_folder = "data/pico-banana" ,
               ceph_folder=None,
               ceph_config=None)


UniworldEdit  = dict(type=ImageEditDataset,
               image_size=image_size,
               image_process=image_process,
               data_path='data/UniWorld-V1-new/UniworldEdit.json',
               image_folder = "data/UniWorld-V1-new" ,
               ceph_folder=None,
               ceph_config=None)



GPT4oEdit_hqedit  = dict(type=ImageEditDataset,
               image_size=image_size,
               image_process=image_process,
               data_path='data/GPT-Image-Edit-1.5M/gpt-edit/hqedit/hqedit.json',
               image_folder = "data/GPT-Image-Edit-1.5M" ,
               ceph_folder=None,
               ceph_config=None)

GPT4oEdit_omniedit  = dict(type=ImageEditDataset,
               image_size=image_size,
               image_process=image_process,
               data_path='data/GPT-Image-Edit-1.5M/gpt-edit/omniedit/omniedit.json',
               image_folder = "data/GPT-Image-Edit-1.5M" ,
               ceph_folder=None,
               ceph_config=None)

GPT4oEdit_ultraedit = dict(type=ImageEditDataset,
               image_size=image_size,
               image_process=image_process,
               data_path='data/GPT-Image-Edit-1.5M/gpt-edit/ultraedit/ultraedit.json',
               image_folder = "data/GPT-Image-Edit-1.5M" ,
               ceph_folder=None,
               ceph_config=None)

Reason_edit = dict(type=ImageEditDataset,
               image_size=image_size,
               image_process=image_process,
               data_path='data/unireason/reson_edit.json',
               image_folder = "data/unireason/edit" ,
               ceph_folder=None,
               ceph_config=None)

Omnigen_edit = dict(type=ImageEditDataset,
               image_size=image_size,
               image_process=image_process,
               data_path='data/X2I2/Omnigen.json',
               image_folder = "data/X2I2/images" ,
               ceph_folder=None,
               ceph_config=None)

nhr_edit_1 = dict(type=ImageEditDataset,
               image_size=image_size,
               image_process=image_process,
               data_path='data/NHR-Edit/NHR-Edit_1.json',
               image_folder = "data/NHR-Edit" ,
               ceph_folder=None,
               ceph_config=None)

nhr_edit_2 = dict(type=ImageEditDataset,
               image_size=image_size,
               image_process=image_process,
               data_path='data/NHR-Edit-part2/NHR-Edit_2.json',
               image_folder = "data/NHR-Edit-part2" ,
               ceph_folder=None,
               ceph_config=None)

dataset = dict(
    type=ConcatDataset,
    datasets=[Open4oEdit,Share4oEdit,nano150Edit,picobananaEdit,UniworldEdit,GPT4oEdit_hqedit,GPT4oEdit_omniedit,GPT4oEdit_ultraedit,Reason_edit,Omnigen_edit,nhr_edit_1,nhr_edit_2],
)

train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    pin_memory=True,
    dataset=dataset,
    sampler=dict(type=InfiniteSampler, shuffle=True),
    collate_fn=dict(type=collate_func_img2img_txt_dynamic)
)
