from src.datasets.collate_functions import (collate_func_img2img_txt_dynamic,
                                            collate_func_gen_txt_dynamic, CollateConcat)
from mmengine.config import read_base
from src.datasets.samplers.multi_source_sampler import MultiSourceSampler, MultiSourceBatchSampler

from xtuner.dataset import ConcatDataset


with read_base():
    from .processors import image_size, image_process
    from .t2i_pretrain import dataset as t2i_pretrain_dataset
    from .edit_pretrain import dataset as edit_pretrain_dataset


dataset = dict(
    type=ConcatDataset,
    datasets=[edit_pretrain_dataset, t2i_pretrain_dataset]
)

group_keys = ['image2image', 'text2image']
repeats = [1, 3] # the radio between editing and generation task 
batch_sizes = [4, 4]
batch_size = sum([repeat * batch_size for repeat, batch_size in zip(repeats, batch_sizes)]) // sum(repeats)


train_dataloader = dict(
    batch_size=batch_size,
    num_workers=4,
    prefetch_factor=1,
    persistent_workers=False,
    pin_memory=True,
    dataset=dataset,
    sampler=dict(type=MultiSourceSampler,
                 repeats=repeats,
                 batch_sizes=batch_sizes,  # fixed batch size for all sources
                 shuffle=True),
    batch_sampler=dict(type=MultiSourceBatchSampler,
                       repeats=repeats,
                       batch_sizes=batch_sizes,
                       ),
    collate_fn=dict(type=CollateConcat,
                    collate_fns=[dict(type=collate_func_img2img_txt_dynamic),
                                 dict(type=collate_func_gen_txt_dynamic),
                                 ],
                    keys=group_keys
                    )
)
