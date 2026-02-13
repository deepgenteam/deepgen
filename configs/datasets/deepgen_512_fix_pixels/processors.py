from transformers import AutoImageProcessor, AutoTokenizer

IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)


qwen2_5_vl_model_name_or_path = "model_zoo/Qwen2.5-VL-3B-Instruct"

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

tokenizer_kwargs = dict(add_special_tokens=True)

pad_index = 0
image_size = 512
image_process = 'fix_pixels'
#######################################################################
#            PART 2  Model & Tokenizer & Image Processor              #
#######################################################################
tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=qwen2_5_vl_model_name_or_path,
    trust_remote_code=True,
    padding_side='right')


image_processor = dict(type=AutoImageProcessor.from_pretrained,
                       pretrained_model_name_or_path=qwen2_5_vl_model_name_or_path)
