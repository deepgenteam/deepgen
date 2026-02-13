from src.datasets.collate_functions import collate_func_gen_txt_dynamic
from mmengine.config import read_base
from mmengine.dataset import InfiniteSampler
from xtuner.dataset import ConcatDataset
from src.datasets.text2image.blip3_o import BLIP3oDataset



with read_base():
    from .processors import image_size, image_process
    from .redcaps5m import dataset as redcaps5m_datasets
    from .laion6m import dataset as laion6m_dataset
    from .text2image2m import dataset as text2image2m_dataset
    from .megalith10m import dataset as megalith10m_dataset
    from .cc12m import dataset as cc12m_dataset





dataset = dict(
    type=ConcatDataset,
    datasets=[redcaps5m_datasets, laion6m_dataset, text2image2m_dataset, megalith10m_dataset,cc12m_dataset],
)


train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    pin_memory=True,
    dataset=dataset,
    sampler=dict(type=InfiniteSampler, shuffle=True),
    collate_fn=dict(type=collate_func_gen_txt_dynamic)
)

