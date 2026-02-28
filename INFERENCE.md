## INFERENCE
Please download our released model weights first from 🤗[deepgenteam/DeepGen-1.0](https://huggingface.co/deepgenteam/DeepGen-1.0). It is recommended to use the following command to download the checkpoints
```bash
# pip install -U "huggingface_hub[cli]"
huggingface-cli download deepgenteam/DeepGen-1.0  --local-dir checkpoints --repo-type model

# Merge zip
cat DeepGen_CKPT.zip.part-* > DeepGen_CKPT.zip
# Unzip DeepGen checkpoints
unzip DeepGen_CKPT.zip
```

```text
checkpoints/
├── DeepGen_CKPT
    ├──Pretrain├──iter_200000.pth
    ├── SFT├──iter_400000.pth
    ├──RL├──MR-GDPO_final.pt

```

the `/path/to/your/ckpt` can be both `.pth folder` or `.pt file`, if you want only final model state please download `model.pt` directly in 🤗[deepgenteam/DeepGen-1.0](https://huggingface.co/deepgenteam/DeepGen-1.0) , it is same as `MR-GDPO_final.pt`

# Text-to-Image

```shell
export PYTHONPATH=.
python scripts/text2image.py
             --checkpoint /path/to/your/ckpt \
             --prompt "a photo of a blue pizza and a yellow baseball glove" \
             --output /path_to_save_result \
             --height 512 --width 512 \
             --seed 42
```
# Image-to-Image(Editing)

```shell
export PYTHONPATH=.
python scripts/image2image.py
             --checkpoint /path/to/your/ckpt \
             --prompt "Using the red color, draw one continuous path from the green start to the red end along walkable white cells only. Do not cross walls." \
             --src_img UniREditBench/original_image/maze/1.png \
             --output /path_to_save_result \
             --height 512 --width 512 \
             --seed 42
```

# Diffusers Format

We also provide a diffusers-compatible format at 🤗[deepgenteam/DeepGen-1.0-diffusers](https://huggingface.co/deepgenteam/DeepGen-1.0-diffusers). This is a self-contained pipeline that **does not require cloning the DeepGen repository**.

### Installation

```bash
pip install torch diffusers transformers safetensors einops accelerate huggingface_hub
# Flash Attention (recommended)
pip install flash-attn --no-build-isolation
```

### Load Pipeline

```python
import torch
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "deepgenteam/DeepGen-1.0-diffusers",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
pipe.to("cuda")

# Optional: enable CPU offload for GPUs with limited memory (< 24GB)
# pipe.enable_model_cpu_offload()
```

### Text-to-Image

```python
result = pipe(
    prompt="a racoon holding a shiny red apple over its head",
    height=512, width=512,
    num_inference_steps=50,
    guidance_scale=4.0,
    seed=42,
)
result.images[0].save("output.png")
```

### Image Editing

```python
from PIL import Image

source_image = Image.open("input.png").convert("RGB")
result = pipe(
    prompt="Place this guitar on a sandy beach with the sunset in the background.",
    image=source_image,
    height=512, width=512,
    num_inference_steps=50,
    guidance_scale=4.0,
    seed=42,
)
result.images[0].save("edited.png")
```

