import random
import torch
import math
from src.models.sd3_kontext.pipeline_stable_diffusion_3_dynamic import StableDiffusion3Pipeline, calculate_shift
from src.models.sd3_kontext.sd3_hf import StableDiffusion3HF
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3


class StableDiffusion3HFDynamic(StableDiffusion3HF):

    def text2image_loss(self, data_dict):

        pixel_values = [img.to(dtype=self.dtype, device=self.device) for img in data_dict['pixel_values']]
        image_latents = [self.pixels_to_latents(img[None])[0] for img in pixel_values]

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
                max_sequence_length=512,
                lora_scale=None,
            )

        loss_diff = self.diff_loss(model_input=image_latents,
                                   pooled_prompt_embeds=pooled_prompt_embeds,
                                   prompt_embeds=prompt_embeds)

        return loss_diff


    def image2image_loss(self, data_dict):
        pixel_values_src = [[img.to(dtype=self.dtype, device=self.device) for img in ref_imgs]
                            for ref_imgs in data_dict['pixel_values_src']]
        image_latents_src = [[self.pixels_to_latents(img[None])[0] for img in ref_imgs]
                             for ref_imgs in pixel_values_src]

        pixel_values = [img.to(dtype=self.dtype, device=self.device) for img in data_dict['pixel_values']]
        image_latents = [self.pixels_to_latents(img[None])[0] for img in pixel_values]

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
                max_sequence_length=512,
                lora_scale=None,
            )


        loss_diff = self.diff_loss(model_input=image_latents,
                                   pooled_prompt_embeds=pooled_prompt_embeds,
                                   prompt_embeds=prompt_embeds,
                                   cond_intput=image_latents_src)

        return loss_diff


    def diff_loss(self, model_input, pooled_prompt_embeds, prompt_embeds, cond_intput=None):
        # Sample noise that we'll add to the latents
        # import pdb; pdb.set_trace()
        noise = [torch.randn_like(x) for x in model_input]
        bsz = len(model_input)

        u = compute_density_for_timestep_sampling(
            weighting_scheme=self.weighting_scheme,
            batch_size=bsz,
            logit_mean=self.logit_mean,
            logit_std=self.logit_std,
        )

        if self.train_scheduler.use_dynamic_shifting:
            assert self.weighting_scheme == 'logit_normal'
            # follow flux
            # import pdb; pdb.set_trace()
            image_seq_lens = [math.prod(x.shape[-2:]) // self.transformer.patch_size ** 2 for x in model_input]
            mu = calculate_shift(
                torch.tensor(image_seq_lens, dtype=self.dtype, device=self.device),
                self.train_scheduler.config.get("base_image_seq_len", 256),
                self.train_scheduler.config.get("max_image_seq_len", 4096),
                self.train_scheduler.config.get("base_shift", 0.5),
                self.train_scheduler.config.get("max_shift", 1.15)
            )

            if self.train_scheduler.config.time_shift_type == "exponential":
                shift = torch.exp(mu)
            elif self.train_scheduler.config.time_shift_type == "linear":
                shift = mu
            else:
                raise NotImplementedError

            sigmas = u.to(dtype=self.dtype, device=self.device)
            sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
            timesteps = sigmas * self.train_scheduler.num_train_timesteps
            sigmas = sigmas.view(-1, 1, 1, 1)

        else:
            # Sample a random timestep for each image
            # for weighting schemes where we sample timesteps non-uniformly
            indices = (u * self.train_scheduler.config.num_train_timesteps).long()
            timesteps = self.train_scheduler.timesteps[indices].to(device=self.device)

            # Add noise according to flow matching.
            # zt = (1 - texp) * x + texp * z1
            sigmas = self.get_sigmas(timesteps, n_dim=model_input[0].ndim + 1)

        noisy_model_input = [(1.0 - x) * y + x * z  for x, y, z in zip(sigmas, model_input, noise)]

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
        # target = noise - model_input
        target = [x - y for x, y in zip(noise, model_input)]

        loss = [(x.float() * (y.float() - z.float()) ** 2).mean() for x, y, z in zip(weighting, model_pred, target)]
        loss = sum(loss) / len(loss)

        return loss



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
            max_sequence_length=512,
            lora_scale=None,
        )

        pipeline.set_progress_bar_config(disable=not progress_bar)

        if pixel_values_src is not None:
            pixel_values_src = [[img.to(dtype=self.dtype, device=self.device) for img in ref_imgs]
                                for ref_imgs in pixel_values_src]
            cond_latents = [[self.pixels_to_latents(img[None])[0] for img in ref_imgs]
                            for ref_imgs in pixel_values_src]
            cond_latents = cond_latents * 2
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


def resize_image(x, image_size, unit_image_size=32):
    w, h = x.size
    if w >= h and w >= image_size:
        target_w = image_size
        target_h = h * (target_w / w)
        target_h = math.ceil(target_h / unit_image_size) * unit_image_size

    elif h >= w and h >= image_size:
        target_h = image_size
        target_w = w * (target_h / h)
        target_w = math.ceil(target_w / unit_image_size) * unit_image_size

    else:
        target_h = math.ceil(h / unit_image_size) * unit_image_size
        target_w = math.ceil(w / unit_image_size) * unit_image_size

    x = x.resize(size=(target_w, target_h))

    return x


if __name__ == "__main__":
    import os
    import argparse
    from glob import glob
    from mmengine.config import Config
    from einops import rearrange
    from PIL import Image
    import numpy as np
    from xtuner.model.utils import guess_load_checkpoint
    from xtuner.registry import BUILDER

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

        if os.path.isdir(args.image):
            ref_images = glob(f"{args.image}/*")
            ref_images = [Image.open(path) for path in ref_images]
        else:
            ref_images = [Image.open(args.image)]

        ref_images = [resize_image(img, max(args.width, args.height), 32) for img in ref_images]

        if len(ref_images) == 1:
            width, height = ref_images[0].size
        else:
            width, height = args.width, args.height

        pixel_values_src = [torch.from_numpy(np.array(img)).to(dtype=model.dtype, device=model.device)
                            for img in ref_images]
        pixel_values_src = [rearrange(img, 'h w c -> c h w') for img in pixel_values_src]
        pixel_values_src = [2 * (img / 255) - 1 for img in pixel_values_src]

        pixel_values_src = [pixel_values_src, ] * bsz
    else:
        width, height = args.width, args.height
        pixel_values_src = None

    samples = model.generate(prompt=prompt, cfg_prompt=cfg_prompt, pixel_values_src=pixel_values_src,
                             cfg_scale=args.cfg_scale, num_steps=args.num_steps,
                             generator=generator, height=height, width=width)


    samples = rearrange(samples, '(m n) c h w -> (m h) (n w) c', m=args.grid_size, n=args.grid_size)
    samples = torch.clamp(
        127.5 * samples + 128.0, 0, 255).to("cpu", dtype=torch.uint8).numpy()

    Image.fromarray(samples).save(args.output)
