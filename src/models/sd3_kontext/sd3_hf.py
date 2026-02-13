import random
import torch
import torch.nn as nn
import torch.distributed as dist
from copy import deepcopy
from torch.nn.modules.module import T
from xtuner.registry import BUILDER
from mmengine.logging import print_log
from xtuner.model.utils import guess_load_checkpoint
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
from peft import LoraConfig
from src.models.sd3_kontext.pipeline_stable_diffusion_3 import StableDiffusion3Pipeline
from mmengine.model import BaseModel



class StableDiffusion3HF(BaseModel):
    def __init__(self,
                 text_encoder,
                 text_encoder_2,
                 text_encoder_3,
                 transformer,
                 train_scheduler,
                 test_scheduler,
                 vae,
                 tokenizer,
                 tokenizer_2,
                 tokenizer_3,
                 pretrained_pth=None,
                 use_activation_checkpointing=True,
                 lora_modules=None,  # ["to_k", "to_q", "to_v", "to_out.0"],
                 lora_rank=8,
                 lora_alpha=8,
                 freeze_transformer=True,
                 unconditional=0.1,
                 weighting_scheme='none',
                 logit_mean=0.0,
                 logit_std=1.0,
                 ema_cfg=None,
                 ):
        super().__init__()
        self.use_activation_checkpointing = use_activation_checkpointing
        self.text_encoder = BUILDER.build(text_encoder)
        self.text_encoder_2 = BUILDER.build(text_encoder_2)
        self.text_encoder_3 = BUILDER.build(text_encoder_3)
        self.text_encoder.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)
        self.text_encoder_3.requires_grad_(False)

        self.tokenizer = BUILDER.build(tokenizer)
        self.tokenizer_2 = BUILDER.build(tokenizer_2)
        self.tokenizer_3 = BUILDER.build(tokenizer_3)

        self.unconditional = unconditional

        self.train_scheduler = BUILDER.build(train_scheduler)
        self.test_scheduler = BUILDER.build(test_scheduler)

        self.transformer = BUILDER.build(transformer)
        if freeze_transformer:
            self.transformer.requires_grad_(False)
        self.freeze_transformer = freeze_transformer

        self.vae = BUILDER.build(vae)
        self.vae.requires_grad_(False)

        self.weighting_scheme = weighting_scheme
        self.logit_mean = logit_mean
        self.logit_std = logit_std

        if use_activation_checkpointing:
            self.gradient_checkpointing_enable()

        if lora_modules is not None:
            assert self.freeze_transformer
            # now we will add new LoRA weights the transformer layers
            transformer_lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                init_lora_weights="gaussian",
                target_modules=lora_modules,
            )
            self.transformer.add_adapter(transformer_lora_config)

        if pretrained_pth is not None:
            pretrained_state_dict = guess_load_checkpoint(pretrained_pth)
            info = self.load_state_dict(pretrained_state_dict, strict=False)
            print_log(f'Load pretrained weight from {pretrained_pth}')


        self.ema_cfg = ema_cfg
        if ema_cfg is not None:
            self.ema = nn.ModuleDict()
            self.ema.steps = 0
            assert not self.freeze_transformer
            self.ema.update(dict(transformer=deepcopy(self.transformer)))
            self.ema.requires_grad_(False)   # parameters in ema are not learnable

            if 'checkpoint' in ema_cfg:
                ema_state_dict = guess_load_checkpoint(ema_cfg['checkpoint'])
                info = self.ema.load_state_dict(ema_state_dict, strict=False)
                print_log(f"Load ema weight from {ema_cfg['checkpoint']}")

    @torch.no_grad()
    def ema_step(self, ):
        if self.ema_cfg is None:
            return

        steps = self.ema.steps
        update_interval = self.ema_cfg.get('update_interval', 1)
        save_interval = self.ema_cfg.get('save_interval', 1)
        momentum = self.ema_cfg.get('momentum', 0.99)

        if steps % update_interval == 0 and steps > 0:
            for ema_param, base_param in zip(self.ema.transformer.parameters(), self.transformer.parameters()):
                ema_param.data.lerp_(base_param.data.detach(), 1.0 - momentum)

        if steps % save_interval == 0 and steps > 0:
            is_ddp = dist.is_available() and dist.is_initialized()
            is_primary_proc = (not is_ddp) or dist.get_rank() == 0
            print(f"steps: {steps}, rank: {dist.get_rank()}, is_ddp:{is_ddp}, is_primary_proc: {is_primary_proc}.", flush=True)
            if is_primary_proc:
                save_path = self.ema_cfg.get('save_path')
                torch.save(self.ema.state_dict(), save_path)
            if is_ddp:
                dist.barrier()

        self.ema.steps = self.ema.steps + 1


    def gradient_checkpointing_enable(self):
        self.activation_checkpointing_enable()

    def activation_checkpointing_enable(self):
        self.transformer.enable_gradient_checkpointing()

    def gradient_checkpointing_disable(self):
        self.activation_checkpointing_disable()

    def activation_checkpointing_disable(self):
        self.transformer.disable_gradient_checkpointing()

    @property
    def device(self):
        return self.transformer.device

    @property
    def dtype(self):
        return self.transformer.dtype

    def train(self: T, mode: bool = True) -> T:
        super().train(mode=mode)
        self.vae.train(mode=False)
        self.text_encoder.train(mode=False)
        self.text_encoder_2.train(mode=False)
        self.text_encoder_3.train(mode=False)
        if not mode:
            self.gradient_checkpointing_disable()

        return self

    def state_dict(self, *args, **kwargs) -> dict:
        state_dict = super().state_dict(*args, **kwargs)
        return {k: v for k, v in state_dict.items() if k.startswith('transformer.')}

    @torch.no_grad()
    def pixels_to_latents(self, x):
        z = self.vae.encode(x).latent_dist.sample()
        z = (z - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        return z

    @torch.no_grad()
    def latents_to_pixels(self, z):
        z = (z / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        x_rec = self.vae.decode(z).sample
        return x_rec

    def forward(self, data, data_samples=None, mode='loss'):
        if mode == 'loss':
            self.ema_step()
            return self.compute_loss(data_dict=data)
        else:
            raise NotImplementedError

    def compute_loss(self, data_dict):
        losses = {}
        for data_type in ['text2image', 'image2image']:
            if data_type in data_dict:
                losses[f'loss_{data_type}'] = getattr(self, f'{data_type}_loss')(data_dict[data_type])
        if len(losses) == 0:
            if 'pixel_values_src' in data_dict:
                losses[f'loss_image2image'] = self.image2image_loss(data_dict)
            else:
                losses[f'loss_text2image'] = self.text2image_loss(data_dict)

        return losses

    def text2image_loss(self, data_dict):

        # obtain image latents
        if 'image_latents' in data_dict:
            image_latents = data_dict['image_latents'].to(dtype=self.dtype, device=self.device)
        else:
            pixel_values = data_dict['pixel_values'].to(dtype=self.dtype, device=self.device)
            image_latents = self.pixels_to_latents(pixel_values)

        texts = ['' if random.uniform(0, 1) < self.unconditional else text
                 for text in data_dict['texts']]

        pipeline = StableDiffusion3Pipeline(
            transformer=None,
            scheduler=None,
            vae=None,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            text_encoder_2=self.text_encoder_2,
            tokenizer_2=self.tokenizer_2,
            text_encoder_3=self.text_encoder_3,
            tokenizer_3=self.tokenizer_3,
        )

        with torch.no_grad():
            (
                prompt_embeds,
                _,
                pooled_prompt_embeds,
                _,
            ) = pipeline.encode_prompt(
                prompt=texts,
                prompt_2=None,
                prompt_3=None,
                negative_prompt=None,
                negative_prompt_2=None,
                negative_prompt_3=None,
                do_classifier_free_guidance=False,
                device=self.device,
                clip_skip=None,
                num_images_per_prompt=1,
                max_sequence_length=256,
                lora_scale=None,
            )

        loss_diff = self.diff_loss(model_input=image_latents,
                                   pooled_prompt_embeds=pooled_prompt_embeds,
                                   prompt_embeds=prompt_embeds)

        return loss_diff


    def image2image_loss(self, data_dict):

        pixel_values_src = data_dict['pixel_values_src'].to(dtype=self.dtype, device=self.device)
        image_latents_src = self.pixels_to_latents(pixel_values_src)

        pixel_values = data_dict['pixel_values'].to(dtype=self.dtype, device=self.device)
        image_latents = self.pixels_to_latents(pixel_values)

        pipeline = StableDiffusion3Pipeline(
            transformer=None,
            scheduler=None,
            vae=None,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            text_encoder_2=self.text_encoder_2,
            tokenizer_2=self.tokenizer_2,
            text_encoder_3=self.text_encoder_3,
            tokenizer_3=self.tokenizer_3,
        )

        with torch.no_grad():
            (
                prompt_embeds,
                _,
                pooled_prompt_embeds,
                _,
            ) = pipeline.encode_prompt(
                prompt=data_dict['texts'],
                prompt_2=None,
                prompt_3=None,
                negative_prompt=None,
                negative_prompt_2=None,
                negative_prompt_3=None,
                do_classifier_free_guidance=False,
                device=self.device,
                clip_skip=None,
                num_images_per_prompt=1,
                max_sequence_length=256,
                lora_scale=None,
            )


        loss_diff = self.diff_loss(model_input=image_latents,
                                   pooled_prompt_embeds=pooled_prompt_embeds,
                                   prompt_embeds=prompt_embeds,
                                   cond_intput=image_latents_src)

        return loss_diff


    @torch.no_grad()
    def generate(self,
                 prompt,
                 cfg_prompt,
                 pixel_values_src=None,
                 cfg_scale=4.5,
                 num_steps=50,
                 generator=None,
                 height=512,
                 width=512,
                 progress_bar=True):

        pipeline = StableDiffusion3Pipeline(
            transformer=self.transformer,
            scheduler=self.test_scheduler,
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            text_encoder_2=self.text_encoder_2,
            tokenizer_2=self.tokenizer_2,
            text_encoder_3=self.text_encoder_3,
            tokenizer_3=self.tokenizer_3,
        )
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = pipeline.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            prompt_3=None,
            negative_prompt=cfg_prompt,
            negative_prompt_2=None,
            negative_prompt_3=None,
            do_classifier_free_guidance=True,
            device=self.device,
            clip_skip=None,
            num_images_per_prompt=1,
            max_sequence_length=256,
            lora_scale=None,
        )

        pipeline.set_progress_bar_config(disable=not progress_bar)

        if pixel_values_src is not None:
            cond_latents = self.pixels_to_latents(pixel_values_src)
            cond_latents = torch.cat([cond_latents] * 2)
        else:
            cond_latents = None

        samples = pipeline(
            height=height,
            width=width,
            guidance_scale=cfg_scale,
            num_inference_steps=num_steps,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            generator=generator,
            output_type='latent',
            cond_latents=cond_latents,
        ).images.to(self.dtype)

        return self.latents_to_pixels(samples)

    def diff_loss(self, model_input, pooled_prompt_embeds, prompt_embeds, cond_intput=None):
        # Sample noise that we'll add to the latents
        noise = torch.randn_like(model_input)
        bsz = model_input.shape[0]

        # Sample a random timestep for each image
        # for weighting schemes where we sample timesteps non-uniformly
        u = compute_density_for_timestep_sampling(
            weighting_scheme=self.weighting_scheme,
            batch_size=bsz,
            logit_mean=self.logit_mean,
            logit_std=self.logit_std,
        )
        indices = (u * self.train_scheduler.config.num_train_timesteps).long()
        timesteps = self.train_scheduler.timesteps[indices].to(device=model_input.device)

        # Add noise according to flow matching.
        # zt = (1 - texp) * x + texp * z1
        sigmas = self.get_sigmas(timesteps, n_dim=model_input.ndim)
        noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

        # Predict the noise residual
        model_pred = self.transformer(
            hidden_states=noisy_model_input,
            cond_hidden_states=cond_intput,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            timestep=timesteps,
            return_dict=False,
        )[0]

        # these weighting schemes use a uniform timestep sampling
        # and instead post-weight the loss
        weighting = compute_loss_weighting_for_sd3(weighting_scheme=self.weighting_scheme, sigmas=sigmas)

        # flow matching loss
        target = noise - model_input

        # Compute regular loss.
        loss = torch.mean(
            (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
            1,
        )
        loss = loss.mean()

        return loss

    def get_sigmas(self, timesteps, n_dim=4):
        sigmas = self.train_scheduler.sigmas.to(device=self.device, dtype=self.dtype)
        schedule_timesteps = self.train_scheduler.timesteps.to(self.device)
        timesteps = timesteps.to(self.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma


if __name__ == "__main__":
    import argparse
    from mmengine.config import Config
    from einops import rearrange
    from PIL import Image
    import numpy as np
    import torch.nn.functional as F

    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='log file path.')
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--prompt", type=str, default='a dog on the left and a cat on the right')
    parser.add_argument("--cfg_prompt", type=str, default='')
    parser.add_argument("--cfg_scale", type=float, default=3.5)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grid_size", type=int, default=2)
    parser.add_argument('--output', type=str, default='output.jpg')

    args = parser.parse_args()
    config = Config.fromfile(args.config)
    model = BUILDER.build(config.model).cuda().bfloat16().eval()

    if args.checkpoint is not None:
        print(f"Load checkpoint: {args.checkpoint}", flush=True)
        checkpoint = guess_load_checkpoint(args.checkpoint)
        info = model.load_state_dict(checkpoint, strict=False)

    generator = torch.Generator(device=model.device).manual_seed(args.seed)
    # repeat
    bsz = args.grid_size ** 2

    prompt = [args.prompt] * bsz
    cfg_prompt = [args.cfg_prompt] * bsz

    if args.image is not None:
        image = Image.open(args.image)
        img_w, img_h = image.size
        image = image.resize(size=(args.height, args.width))
        pixel_values_src = torch.from_numpy(np.array(image)).to(dtype=model.dtype, device=model.device)
        pixel_values_src = rearrange(pixel_values_src, 'h w c -> c h w')[None]
        pixel_values_src = 2 * (pixel_values_src / 255) - 1
        pixel_values_src = pixel_values_src.expand(bsz, -1, -1, -1)
    else:
        pixel_values_src = None

    samples = model.generate(prompt=prompt, cfg_prompt=cfg_prompt, pixel_values_src=pixel_values_src,
                             cfg_scale=args.cfg_scale, num_steps=args.num_steps,
                             generator=generator, height=args.height, width=args.width)

    if pixel_values_src is not None:
        samples = F.interpolate(samples, size=(img_h, img_w), mode='bilinear')

    samples = rearrange(samples, '(m n) c h w -> (m h) (n w) c', m=args.grid_size, n=args.grid_size)
    samples = torch.clamp(
        127.5 * samples + 128.0, 0, 255).to("cpu", dtype=torch.uint8).numpy()

    Image.fromarray(samples).save(args.output)
