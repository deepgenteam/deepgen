

<p align="center">
  <img src="figure/logo.jpg" alt="DeepGen" width="450"/ hei>
</p>

<h1 align="center">DeepGen 1.0: A Lightweight Unified Multimodal
Model for Advancing Image Generation and Editing
</h1>

<p align="center">
Shanghai Innovation Institute, DeepGen Team
</p>

<p align="center">
<a href="http://arxiv.org/abs/2602.12205">
<img src='https://img.shields.io/badge/arXiv-DeepGen-blue' alt='Paper PDF'></a>

<a href="https://deepgenteam.github.io/">
<img src='https://img.shields.io/badge/Website-project page-orange' alt='Project Page'></a>

<a href="https://github.com/deepgenteam/deepgen_rl">
<img src='https://img.shields.io/badge/GitHub-DeepGen--RL-black' alt='DeepGen RL'></a>

<a href="https://huggingface.co/deepgenteam/DeepGen-1.0">
<img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-DeepGen-yellow' alt='Model CkPT'></a>

<a href="https://huggingface.co/datasets/DeepGenTeam/DeepGen-1.0">
<img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Data Coming Soon-yellow' alt='Data'></a>
</p>


## 🔥 News
- **Feb 13, 2026:** We released **DeepGen 1.0**, Pre-traning, Supervised Fine-Tuning and Reinforcement Learning checkpoints can be found in [Huggingface](https://huggingface.co/deepgenteam/DeepGen-1.0), support both T2I generation and image editing.
- **Feb 13, 2026:** We released the training code support Pre-traning, Supervised Fine-Tuning, Reinforcement Learning(deepgen_rl) and evaluation code support wide range of benchmarks.
- **Feb 13, 2026:** We released the **DeepGen 1.0** technical report on [Arxiv](http://arxiv.org/abs/2602.12205)




## ✨ Introduction
**Broader Scenario and Dimension Coverage**
We propose DeepGen 1.0, a lightweight unified multimodal model with only 5B parameters (3B VLM + 2B DiT). It integrates five core capabilities: general image generation, general image editing, reasoning image generation, reasoning image editing, and text rendering—within a single model. Across multiple authoritative benchmarks, DeepGen 1.0 is competitive with or surpassing the state-of-the-art unified multimodal models that are 3× to 16× larger, achieving comprehensive performance, demonstrating that massive scaling is not the sole path to high-performance multimodal generation.

<p align="center"><img src="figure/bubble_chart.png" width="95%"></p>




## 🧠 Method
Our core observation is that a lightweight model, when empowered by synergistic architecture design and data-centric training strategies, can achieve comprehensive capabilities competitive with or even surpassing much larger counterparts.
To overcome the limitations of lightweight models in semantic understanding and fine-grained control, we introduce **Stacked Channel Bridging (SCB)**, a deep alignment framework that extracts hierarchical features from multiple VLM layers and fuses them with learnable "think tokens" to provide the generative backbone with structured, reasoning-rich guidance.
We further design a data-centric training strategy spanning three progressive stages: (1) **Alignment Pre-training** on large-scale image-text pairs and editing triplets to synchronize VLM and DiT representations, (2) **Joint Supervised Fine-tuning** on a high-quality mixture of generation, editing, and reasoning tasks to foster omni-capabilities, and (3) **Reinforcement Learning with MR-GRPO**, which leverages a mixture of reward functions and supervision signals, resulting in substantial gains in generation quality and alignment with human preferences, while maintaining stable training progress and avoiding visual artifacts.

<p align="center"><img src="figure/arch.png" width="80%"></p>


## 💻 Train & Eval

### Set up environment
```bash
git clone https://github.com/deepgenteam/deepgen.git
cd deepgen
conda create -n deepgen python=3.12 -y
conda activate deepgen
pip install -r requirements.txt
pip install flash_attn==2.8.3 --no-build-isolation
pip install xtuner==0.2.0
pip install transformers==4.56.1
pip install triton==2.3.0
pip install -U opencv-python-headless
```
### Data Prepare
Please See [DATA](DATA.md) for more details. We provide a detailed description of the data download and usage procedures for both the Pre-traning stage and the Supervised Fine-Tuning stage.

### Inference

<details>
<summary><b>Diffusers</b> (click to expand)</summary>

We provide a diffusers-compatible format at 🤗[deepgenteam/DeepGen-1.0-diffusers](https://huggingface.co/deepgenteam/DeepGen-1.0-diffusers).

```bash
pip install torch diffusers transformers safetensors einops accelerate huggingface_hub
pip install flash-attn --no-build-isolation  # optional but recommended
```

**Text-to-Image:**

```python
import torch
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "deepgenteam/DeepGen-1.0-diffusers",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
pipe.to("cuda")

result = pipe(
    prompt="a photo of a blue pizza and a yellow baseball glove",
    height=512, width=512,
    num_inference_steps=50,
    guidance_scale=4.0,
    seed=42,
)
result.images[0].save("output.png")
```

**Image Editing:**

```python
from PIL import Image

result = pipe(
    prompt="Place this guitar on a sandy beach with the sunset in the background.",
    image=Image.open("guitar.png"),
    negative_prompt="blurry, low quality, low resolution, distorted, deformed, broken content, missing parts, damaged details, artifacts, glitch, noise, pixelated, grainy, compression artifacts, bad composition, wrong proportion, incomplete editing, unfinished, unedited areas.",
    height=512, width=512,
    num_inference_steps=50,
    guidance_scale=4.0,
    seed=42,
)
result.images[0].save("edited.png")
```

</details>

Please refer to [INFERENCE](INFERENCE.md) for more details, including the native pipeline usage.

### Train

See [TRAIN](TRAIN.md) for more details. We provide a detailed description of the model and training configs for both the Pre-traning stage and the Supervised Fine-Tuning stage.

### Eval
We provide the scripts for evaluating wide range of T2I and Editing benchmarks. Please See [EVAL](EVAL.md) for more details.

## 📊 Benchmarks

### 1. General Image Generation
| Model                 | Params      | Geneval ↑   | DPGBench ↑   | UniGenBench ↑ |
| --------------------- | ----------- | ----------- | ------------ | ------------- |
| OmniGen2                 | 3B + 4B         | 0.80         | 83.57         | 63.09        |
| BAGEL                 | 14B         | 0.82        | 85.10        | 61.53         |
| X-Omni                 | 7B + 12B         | 0.83         | 87.65🥉        | 53.77         |
| Lumina-DiMOO                 | 8B         | 0.88🥇          | 86.04        | 71.12         |
| Hunyuan-Image-3.0     | 80B         | 0.72        | 86.10        | —             |
| Qwen-Image            | 7B + 20B    | 0.87 🥈     | 88.32 🥇     | 78.81 🥇      |
| LongCat-Image         | 7B + 6B     | 0.87 🥈     | 86.80        | —             |
| Z-Image-Turbo         | 4B + 6B     | 0.84        | 85.15        | 71.40         |
| GLM-Image             | 9B + 7B     | —           | 84.78        | —             |
| **DeepGen 1.0 (SFT)** | **3B + 2B** | 0.86 🥉 | 87.05    | 74.18 🥉  |
| **DeepGen 1.0 (RL)**  | **3B + 2B** | 0.87 🥈 | 87.90 🥈 | 75.74 🥈  |



### 2. General Image Editing

| Model | Params | GEdit-EN ↑ | ImgEdit ↑ |
| :--- | :--- | :--- | :--- |
| BAGEL | 14B | 6.52 | 3.20 |
| Qwen-Image-Edit [2509] | 7B + 20B | 7.54 🥈 | 4.35 🥈 |
| LongCat-Image-Edit | 7B + 6B | 7.60 🥇 | 4.50 🥇 |
| Mammoth2 | 8B + 3B + 2B | 6.60 | 4.06 |
| **DeepGen 1.0 (SFT)** | **3B + 2B** | 7.12 | 4.09 |
| **DeepGen 1.0 (RL)** | **3B + 2B** | 7.17 🥉 | 4.14 🥉 |

### 3. Reasoning Image Generation
| Model | Params | WISE ↑ | T2I-CoREBench ↑ |
| :--- | :--- | :--- | :--- |
| OmniGen2 | 3B + 4B | 0.47 | 36.1 |
| BAGEL | 14B | 0.70 🥉 | 41.1 |
| Hunyuan-Image-3.0 | 80B | 0.57 | 46.0 |
| Qwen-Image | 7B + 20B | 0.62 | 46.3 🥉 |
| LongCat-Image | 7B + 6B | 0.65 | 52.2 🥇 |
| Z-Image-Turbo | 4B + 6B | - | 43.7 |
| **DeepGen 1.0 (SFT)** | **3B + 2B** | 0.72 🥈 | 45.7 |
| **DeepGen 1.0 (RL)** | **3B + 2B** | 0.73 🥇 | 46.5 🥈 |

### 4. Reasoning Image Editing

| Model | Params | RISE ↑ | UniREditBench ↑ |
| :--- | :--- | :--- | :--- |
| OmniGen2 | 3B + 4B | - | 43.4 |
| BAGEL | 14B | 11.9 🥈 | 51.0 |
| Qwen-Image-Edit [2509] | 7B + 20B | 8.9 | 56.5 🥉 |
| **DeepGen 1.0 (SFT)** | **3B + 2B** | 13.3 🥇 | 77.5 🥇 |
| **DeepGen 1.0 (RL)** | **3B + 2B** | 10.8 🥉 | 75.7 🥈 |



## 📧 Contact
dywang24@m.fudan.edu.cn, wjqdev@gmail.com
## 🎨 Quantitative results
<p align="left"><img src="figure/teaser.png" width="80%"></p>

## ⭐ Citation
```bibtex
@article{wang2026deepgen,
  title={DeepGen 1.0: A Lightweight Unified Multimodal Model for Advancing Image Generation and Editing},
  author={Wang, Dianyi and Li, Ruihang and Han, Feng and Ma, Chaofan and Song, Wei and Wang, Siyuan and Wang, Yibin and Xin, Yi and Liu, Hongjian and Zhang, Zhixiong and others},
  journal={arXiv preprint arXiv:2602.12205},
  year={2026}
}
```

## 🙏 Acknowledgement
The project builds upon the following pioneering works:
- [OpenUni](https://arxiv.org/abs/2505.23661): We thank the OpenUni releasing the elegant and concise code and pretrain dataset.
- [UniPic2-SD3.5M-Kontext-2B](https://huggingface.co/Skywork/UniPic2-SD3.5M-Kontext-2B): We use UniPic2-SD3.5M-Kontext-2B as our diffusion module, considering its efficiency and strong performance on both t2i and editing.
- [UnifiedReward-Think](https://github.com/CodeGoat24/UnifiedReward): We use UnifiedReward-Think as our reward model for RL, considering its strong performance.
- [Qwen2.5 VL](https://arxiv.org/abs/2502.13923): We useQwen2.5 VL-3B as our VLM module, considering its efficiency and strong performance on multimodal understanding abilities.
- [BLIP3-o](https://github.com/JiuhaiChen/BLIP3o): We thank the BLIP3-o team for releasing the precious high-quality tuning dataset.
- [OpenGPT-4o-Image](https://arxiv.org/abs/2509.24900): We thank the OpenGPT-4o-Image team for releasing the precious high-quality tuning dataset.
- [ShareGPT-4o-Image](https://arxiv.org/abs/2506.18095): We thank the ShareGPT-4o-Image team for releasing the precious high-quality tuning dataset.
- [Echo-4o](https://arxiv.org/abs/2508.09987): We thank the Echo-4o team for releasing the precious high-quality tuning dataset.
- [OmniGen2](https://github.com/VectorSpaceLab/OmniGen2): We thank the OmniGen2 team for releasing the precious high-quality editing tuning dataset and code.
- [Uniworld-V1](https://github.com/PKU-YuanGroup/UniWorld): We thank the Uniworld team for releasing the precious high-quality tuning dataset and code.
- [Picobanana](https://github.com/apple/pico-banana-400k): We thank the Picobanana team for releasing the precious high-quality editing tuning dataset.
- [Nano-consist](https://huggingface.co/datasets/Yejy53/Nano-consistent-150k): We thank the Nano-consist team for releasing the precious high-quality editing tuning dataset.
- [NHR-edit](https://huggingface.co/datasets/iitolstykh/NHR-Edit): We thank the NHR-edit team for releasing the precious high-quality editing tuning dataset.
- [UniREditBench](https://maplebb.github.io/UniREditBench/): We thank the UniREditBench team for releasing the precious high-quality reason-based editing tuning dataset.
