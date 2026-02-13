import torch
from src.models.sd3_kontext.qwen2_5_vl_sd3_hf_dynamic import Qwen2p5VLStableDiffusion3HF
from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer
from src.models.sd3_kontext.transformer_sd3_dynamic import SD3Transformer2DModel


sd3_5_model_name_or_path = "model_zoo/UniPic2-SD3.5M-Kontext-2B"
qwen2_5_vl_model_name_or_path = "model_zoo/Qwen2.5-VL-3B-Instruct"

tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=qwen2_5_vl_model_name_or_path,
    trust_remote_code=True,
    padding_side='right')

prompt_template = dict(
    IMG_START_TOKEN='<|vision_start|>',
    IMG_END_TOKEN='<|vision_end|>',
    IMG_CONTEXT_TOKEN='<|image_pad|>',
    IMG_START_TOKEN_FOR_GENERATION=False,
    SYSTEM=('<|im_start|>system\n{system}<|im_end|>\n'),
    INSTRUCTION=('<|im_start|>user\n{input}<|im_end|>\n'
                 '<|im_start|>assistant\n'),
    SUFFIX='<|im_end|>',
    SUFFIX_AS_EOS=True,
    SEP='\n',
    STOP_WORDS=['<|im_end|>', '<|endoftext|>'],
    GENERATION='Generate an image: {input}',
    CFG='Generate an image.'
)


model = dict(
    type=Qwen2p5VLStableDiffusion3HF,
    num_queries=128,
    connector=dict(
        hidden_size=2048,
        intermediate_size=11946,
        num_hidden_layers=6,
        _attn_implementation='flash_attention_2',
        num_attention_heads=32, ),
    lmm=dict(type=Qwen2_5_VLForConditionalGeneration.from_pretrained,
             pretrained_model_name_or_path=qwen2_5_vl_model_name_or_path,
             torch_dtype=torch.bfloat16,
             attn_implementation="flash_attention_2", ),
    tokenizer=tokenizer,
    prompt_template=prompt_template,
    freeze_lmm=True,
    transformer=dict(
        type=SD3Transformer2DModel.from_pretrained,
        pretrained_model_name_or_path=sd3_5_model_name_or_path,
        subfolder="transformer",
        torch_dtype=torch.bfloat16),
    test_scheduler=dict(
        type=FlowMatchEulerDiscreteScheduler.from_pretrained,
        pretrained_model_name_or_path=sd3_5_model_name_or_path,
        subfolder="scheduler"),
    train_scheduler=dict(
        type=FlowMatchEulerDiscreteScheduler.from_pretrained,
        pretrained_model_name_or_path=sd3_5_model_name_or_path,
        subfolder="scheduler"),
    vae=dict(
        type=AutoencoderKL.from_pretrained,
        pretrained_model_name_or_path=sd3_5_model_name_or_path,
        subfolder="vae",
        torch_dtype=torch.bfloat16),
    pretrained_pth=None,
    use_activation_checkpointing=False,
    freeze_transformer=True,
)
