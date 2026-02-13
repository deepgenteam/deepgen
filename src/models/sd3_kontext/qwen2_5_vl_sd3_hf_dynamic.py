import random
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import T
import torch.distributed as dist
from mmengine.logging import print_log
from src.models.connector import ConnectorConfig, ConnectorEncoder
from xtuner.model.utils import guess_load_checkpoint
from xtuner.registry import BUILDER
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
from peft import LoraConfig
from src.models.sd3_kontext.pipeline_stable_diffusion_3_dynamic import StableDiffusion3Pipeline, calculate_shift
from mmengine.model import BaseModel
from functools import partial
from six.moves import map, zip
from copy import deepcopy
from einops import rearrange


IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)


def multi_apply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def find_target_linear_names(model, num_lora_modules=-1, lora_namespan_exclude=[], verbose=True):
    linear_cls = torch.nn.modules.Linear
    embedding_cls = torch.nn.modules.Embedding
    lora_module_names = []

    for name, module in model.named_modules():
        if any(ex_keyword in name for ex_keyword in lora_namespan_exclude):
            continue
        if isinstance(module, (linear_cls, embedding_cls)):
            lora_module_names.append(name)
    
    if num_lora_modules > 0:
        lora_module_names = lora_module_names[-num_lora_modules:]
    if verbose:
        print(f"Found {len(lora_module_names)} lora modules: {lora_module_names}")
    return lora_module_names


class Qwen2p5VLStableDiffusion3HF(BaseModel):
    def __init__(self,
                 transformer,
                 train_scheduler,
                 test_scheduler,
                 vae,
                 lmm,
                 tokenizer,
                 prompt_template,
                 connector,
                 num_queries=64,
                 vit_input_size=448,
                 max_length=1024,
                 freeze_lmm=True,
                 freeze_mq=False,
                 res_vit=False,
                 pretrained_pth=None,
                 use_activation_checkpointing=False,
                 lora_modules='auto',  # ["to_k", "to_q", "to_v", "to_out.0"], 'auto'
                 lora_rank=64,
                 lora_alpha=128,
                 freeze_transformer=True,
                 unconditional=0.1,
                 ema_cfg=None,
                 weighting_scheme='none',
                 logit_mean=0.0,
                 logit_std=1.0,
                 ):
        super().__init__()

        self.lmm = BUILDER.build(lmm)
        if freeze_lmm:
            self.lmm.requires_grad_(False)
        self.freeze_lmm = freeze_lmm

        self.transformer = BUILDER.build(transformer)
        if freeze_transformer:
            self.transformer.requires_grad_(False)
        self.freeze_transformer = freeze_transformer
        self.res_vit = res_vit



        self.weighting_scheme = weighting_scheme
        self.logit_mean = logit_mean
        self.logit_std = logit_std

        self.vae = BUILDER.build(vae)
        self.vae.requires_grad_(False)

        self.use_activation_checkpointing = use_activation_checkpointing
        self.tokenizer = BUILDER.build(tokenizer)

        self.prompt_template = prompt_template
        self.vit_input_size = vit_input_size
        self.max_length = max_length
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(prompt_template['IMG_CONTEXT_TOKEN'])
        self.register_buffer('vit_mean', torch.tensor(IMAGE_MEAN), persistent=False)
        self.register_buffer('vit_std', torch.tensor(IMAGE_STD), persistent=False)

        self.num_queries = num_queries
        self.connector = ConnectorEncoder(ConnectorConfig(**connector))


        self.projector_1 = nn.Linear(self.llm.config.hidden_size, self.connector.config.hidden_size)
        self.projector_2 = nn.Linear(self.connector.config.hidden_size, self.transformer.config.pooled_projection_dim)
        self.projector_3 = nn.Linear(self.connector.config.hidden_size, self.transformer.config.joint_attention_dim)

        # zero out
        nn.init.zeros_(self.projector_2.weight)
        nn.init.zeros_(self.projector_3.weight)
        nn.init.zeros_(self.projector_2.bias)
        nn.init.zeros_(self.projector_3.bias)

        self.meta_queries = nn.Parameter(
            torch.zeros(num_queries, self.llm.config.hidden_size))
        nn.init.normal_(self.meta_queries, std=1 / math.sqrt(self.llm.config.hidden_size))

        if freeze_mq:
            self.projector_1.requires_grad_(False)
            self.projector_2.requires_grad_(False)
            self.projector_3.requires_grad_(False)
            self.connector.requires_grad_(False)
            self.meta_queries.requires_grad_(False)
        self.freeze_mq = freeze_mq

        self.unconditional = unconditional

        self.train_scheduler = BUILDER.build(train_scheduler)
        self.test_scheduler = BUILDER.build(test_scheduler)

        if use_activation_checkpointing:
            self.gradient_checkpointing_enable()

        if lora_modules is not None:
            assert self.freeze_lmm
            self.llm.config.tie_word_embeddings = False
            if lora_modules == 'auto':
                lora_modules = find_target_linear_names(self.lmm)
            # now we will add new LoRA weights the transformer layers
            transformer_lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                init_lora_weights="gaussian",
                target_modules=lora_modules,
                lora_dropout=0.05,
            )
            self.lmm.add_adapter(transformer_lora_config)

        if pretrained_pth is not None:
            pretrained_state_dict = guess_load_checkpoint(pretrained_pth)
            info = self.load_state_dict(pretrained_state_dict, strict=False)
            print_log(f'Load pretrained weight from {pretrained_pth}')

        self.ema_cfg = ema_cfg
        if ema_cfg is not None:
            self.ema = nn.ModuleDict()
            self.ema.steps = 0
            if not self.freeze_transformer:
                self.ema.update(dict(transformer=deepcopy(self.transformer)))

            if not self.freeze_mq:
                self.ema.update(dict(projector_1=deepcopy(self.projector_1),
                                     projector_2=deepcopy(self.projector_2),
                                     projector_3=deepcopy(self.projector_3),
                                     connector=deepcopy(self.connector)
                                     )
                                )
                self.ema.register_buffer('meta_queries', deepcopy(self.meta_queries.data))

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
        save_interval = self.ema_cfg.get('save_interval', 1000)
        momentum = self.ema_cfg.get('momentum', 0.99)

        if steps % update_interval == 0 and steps > 0:
            if not self.freeze_mq:
                for ema_param, base_param in zip(self.ema.projector_1.parameters(), self.projector_1.parameters()):
                    ema_param.data.lerp_(base_param.data.detach(), 1.0 - momentum)
                for ema_param, base_param in zip(self.ema.projector_2.parameters(), self.projector_2.parameters()):
                    ema_param.data.lerp_(base_param.data.detach(), 1.0 - momentum)
                for ema_param, base_param in zip(self.ema.projector_3.parameters(), self.projector_3.parameters()):
                    ema_param.data.lerp_(base_param.data.detach(), 1.0 - momentum)
                for ema_param, base_param in zip(self.ema.connector.parameters(), self.connector.parameters()):
                    ema_param.data.lerp_(base_param.data.detach(), 1.0 - momentum)
                self.ema.meta_queries.data.lerp_(self.meta_queries.data.detach(), 1.0 - momentum)

            if not self.freeze_transformer:
                for ema_param, base_param in zip(self.ema.transformer.parameters(), self.transformer.parameters()):
                    ema_param.data.lerp_(base_param.data.detach(), 1.0 - momentum)

        # print(f"steps: {steps}, rank: {dist.get_rank()},", flush=True)

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

    def llm2dit(self, x):
        x = self.connector(self.projector_1(x))
        pooled_out = self.projector_2(x.mean(1))
        seq_out = self.projector_3(x)

        return pooled_out, seq_out

    @property
    def llm(self):
        return self.lmm.language_model

    def gradient_checkpointing_enable(self):
        self.activation_checkpointing_enable()

    def activation_checkpointing_enable(self):
        self.llm.gradient_checkpointing_enable()
        self.transformer.enable_gradient_checkpointing()
        self.connector.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.activation_checkpointing_disable()

    def activation_checkpointing_disable(self):
        self.llm.gradient_checkpointing_disable()
        self.transformer.disable_gradient_checkpointing()
        self.connector.gradient_checkpointing = False

    @property
    def device(self):
        return self.llm.device

    @property
    def dtype(self):
        return self.llm.dtype

    def train(self: T, mode: bool = True) -> T:
        super().train(mode=mode)
        if self.vae is not None:
            self.vae.train(mode=False)
        if not mode:
            self.gradient_checkpointing_disable()

        return self

    def state_dict(self, *args, **kwargs) -> dict:
        state_dict = super().state_dict(*args, **kwargs)
        state_dict = {k: v for k, v in state_dict.items()
                      if 'vae.' not in k and 'lmm.' not in k and 'ema.' not in k}
        return state_dict

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


    def prepare_forward_input(self,
                              query_embeds,
                              input_ids=None,
                              image_embeds=None,
                              image_grid_thw=None,
                              attention_mask=None,
                              past_key_values=None):
        b, l, _ = query_embeds.shape
        assert l > 0
        attention_mask = attention_mask.to(device=self.device, dtype=torch.bool)

        assert l == self.num_queries

        input_ids = torch.cat([input_ids, input_ids.new_zeros(b, l)], dim=1)
        attention_mask = torch.cat([attention_mask, attention_mask.new_ones(b, l)], dim=1)

        position_ids, _ = self.lmm.model.get_rope_index(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=None,
            second_per_grid_ts=None,
            attention_mask=attention_mask,
        )

        # prepare context
        if past_key_values is not None:
            inputs_embeds = query_embeds
            position_ids = position_ids[..., -l:]
        else:
            input_ids = input_ids[:, :-l]   # context input_ids

            if image_embeds is None:
                inputs_embeds = self.llm.get_input_embeddings()(input_ids)
            else:
                inputs_embeds = torch.zeros(*input_ids.shape, self.llm.config.hidden_size,
                                            device=self.device, dtype=self.dtype)
                inputs_embeds[input_ids == self.image_token_id] = \
                    image_embeds.contiguous().view(-1, self.llm.config.hidden_size)
                inputs_embeds[input_ids != self.image_token_id] = self.llm.get_input_embeddings()(
                    input_ids[input_ids != self.image_token_id]
                )

            inputs_embeds = torch.cat([inputs_embeds, query_embeds], dim=1)

        inputs = dict(inputs_embeds=inputs_embeds,
                      attention_mask=attention_mask,
                      position_ids=position_ids,
                      past_key_values=past_key_values)

        return inputs


    @torch.no_grad()
    def get_semantic_features_dynamic(self, pixel_values):
        # e.g., 512 -> 448
        pixel_values = [F.interpolate(p[None], scale_factor=28 / 32, mode='bilinear') for p in pixel_values]
        image_embeds, image_grid_thw = multi_apply(self.get_semantic_features,
                                                   pixel_values, resize=False)
        image_embeds = [x[0] for x in image_embeds]    # a list of embeds
        image_grid_thw = torch.cat(image_grid_thw, dim=0)   # b 3

        return image_embeds, image_grid_thw

    @torch.no_grad()
    def get_semantic_features(self, pixel_values, resize=True):
        # pixel_values: [-1, 1]
        pixel_values = (pixel_values + 1.0) / 2     # [0, 1]
        pixel_values = pixel_values - self.vit_mean.view(1, 3, 1, 1)
        pixel_values = pixel_values / self.vit_std.view(1, 3, 1, 1)

        if resize:
           pixel_values = F.interpolate(pixel_values, size=(self.vit_input_size, self.vit_input_size),
                                        mode='bilinear')
        b, c, h, w = pixel_values.shape

        patch_size = self.lmm.config.vision_config.patch_size
        spatial_merge_size = self.lmm.config.vision_config.spatial_merge_size
        temporal_patch_size = self.lmm.config.vision_config.temporal_patch_size

        pixel_values = pixel_values[:, None].expand(b, temporal_patch_size, c, h, w)

        grid_t = 1
        grid_h, grid_w = h // patch_size, w // patch_size

        pixel_values = pixel_values.view(
            b,
            grid_t,
            temporal_patch_size,
            c,
            grid_h // spatial_merge_size,
            spatial_merge_size,
            patch_size,
            grid_w // spatial_merge_size,
            spatial_merge_size,
            patch_size,
        )

        pixel_values = rearrange(
            pixel_values, 'b t tp c h m p w n q -> (b t h w m n) (c tp p q)')

        image_grid_thw = torch.tensor([(grid_t, grid_h, grid_w)] * b).to(self.device).long()

        image_embeds = self.lmm.visual(pixel_values, grid_thw=image_grid_thw)
        image_embeds = rearrange(image_embeds, '(b l) d -> b l d', b=b)

        return image_embeds, image_grid_thw

    @torch.no_grad()
    def prepare_text2image_prompts(self, texts):
        texts = [self.prompt_template['GENERATION'].format(input=text) for text in texts]
        texts = [self.prompt_template['INSTRUCTION'].format(input=text) for text in texts]

        return self.tokenizer(
            texts, add_special_tokens=True, return_tensors='pt', padding=True, padding_side='left').to(self.device)

    @torch.no_grad()
    def prepare_image2image_prompts(self, texts, num_refs, ref_lens):
        prompts = []
        cnt = 0
        for text, num_ref in zip(texts, num_refs):
            image_tokens = ''
            for _ in range(num_ref):
                image_tokens +=  self.prompt_template['IMG_START_TOKEN'] + \
                                 self.prompt_template['IMG_CONTEXT_TOKEN'] * ref_lens[cnt] + \
                                 self.prompt_template['IMG_END_TOKEN']
                cnt += 1

            prompts.append(self.prompt_template['INSTRUCTION'].format(input=f'{image_tokens}\n{text}'))

        return self.tokenizer(
            prompts, add_special_tokens=True, return_tensors='pt', padding=True, padding_side='left').to(self.device)


    def text2image_loss(self, data_dict):
        # obtain image latents
        if 'image_latents' in data_dict:
            image_latents = data_dict['image_latents']   # .to(dtype=self.dtype, device=self.device)
            image_latents = [x.to(dtype=self.dtype, device=self.device) for x in image_latents]
        else:
            pixel_values = [p.to(dtype=self.dtype, device=self.device) for p in data_dict['pixel_values']]
            image_latents = [self.pixels_to_latents(p[None])[0] for p in pixel_values]

        b = len(image_latents)

        texts = ['' if random.uniform(0, 1) < self.unconditional else text
                 for text in data_dict['texts']]

        text_inputs = self.prepare_text2image_prompts(texts)
        hidden_states = self.meta_queries[None].expand(b, self.num_queries, -1)

        inputs = self.prepare_forward_input(query_embeds=hidden_states, **text_inputs)

        max_length = self.max_length + self.num_queries
        inputs_embeds = inputs['inputs_embeds'][:, -max_length:]
        attention_mask = inputs['attention_mask'][:, -max_length:]
        position_ids = inputs['position_ids'][..., -max_length:]

        output = self.llm(inputs_embeds=inputs_embeds,
                          attention_mask=attention_mask,
                          position_ids=position_ids,
                          return_dict=True)

        # hidden_states = output.last_hidden_state[:, -self.num_queries:]#query only
        hidden_states = output.last_hidden_state
        pooled_out, seq_out = self.llm2dit(hidden_states)

        loss_diff = self.diff_loss(model_input=image_latents,
                                   pooled_prompt_embeds=pooled_out,
                                   prompt_embeds=seq_out)

        return loss_diff


    def image2image_loss(self, data_dict):
        pixel_values_src = data_dict['pixel_values_src']

        num_refs = [len(ref_images) for ref_images in pixel_values_src]

        pixel_values_src = [[img.to(dtype=self.dtype, device=self.device) for img in ref_images]
                            for ref_images in pixel_values_src]
        image_latents_src = [[self.pixels_to_latents(img[None])[0] for img in ref_images]
                             for ref_images in pixel_values_src]
        image_embeds, image_grid_thw = self.get_semantic_features_dynamic(
            [img for ref_images in pixel_values_src for img in ref_images])

        ref_lens = [len(x) for x in image_embeds]

        pixel_values = [p.to(dtype=self.dtype, device=self.device) for p in data_dict['pixel_values']]
        image_latents = [self.pixels_to_latents(p[None])[0] for p in pixel_values]

        b = len(image_latents)
        text_inputs = self.prepare_image2image_prompts(data_dict['texts'], num_refs=num_refs, ref_lens=ref_lens)

        hidden_states = self.meta_queries[None].expand(b, self.num_queries, -1)

        inputs = self.prepare_forward_input(query_embeds=hidden_states,
                                            image_embeds=torch.cat(image_embeds),
                                            image_grid_thw=image_grid_thw,
                                            **text_inputs)


        max_length = self.max_length + max(num_refs) * max(ref_lens) + self.num_queries
        inputs_embeds = inputs['inputs_embeds'][:, -max_length:]
        attention_mask = inputs['attention_mask'][:, -max_length:]
        position_ids = inputs['position_ids'][..., -max_length:]

        output = self.llm(inputs_embeds=inputs_embeds,
                          attention_mask=attention_mask,
                          position_ids=position_ids,
                          return_dict=True)

        # hidden_states = output.last_hidden_state[:, -self.num_queries:] #query only
        hidden_states = output.last_hidden_state

        # if res_vit:
        #     image_embeds=torch.cat(image_embeds)
        #     hidden_states[input_ids == self.image_token_id] = 0.5 * (hidden_states[input_ids == self.image_token_id]) + 0.5 * (image_embeds.contiguous().view(-1, self.llm.config.hidden_size))


        pooled_out, seq_out = self.llm2dit(hidden_states)


        loss_diff = self.diff_loss(model_input=image_latents,
                                   pooled_prompt_embeds=pooled_out,
                                   prompt_embeds=seq_out,
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
        assert len(prompt) == len(cfg_prompt)
        b = len(prompt)

        if pixel_values_src is not None:
            num_refs = [len(ref_images) for ref_images in pixel_values_src]
            pixel_values_src = [[img.to(dtype=self.dtype, device=self.device) for img in ref_imgs]
                                for ref_imgs in pixel_values_src]
            image_embeds, image_grid_thw = self.get_semantic_features_dynamic(
                [img for ref_images in pixel_values_src for img in ref_images])
            ref_lens = [len(x) for x in image_embeds]

            text_inputs = self.prepare_image2image_prompts(prompt + cfg_prompt, num_refs=num_refs*2, ref_lens=ref_lens*2)
            text_inputs.update(image_embeds=torch.cat(image_embeds*2),
                               image_grid_thw=torch.cat([image_grid_thw]*2),)
            cond_latents = [[self.pixels_to_latents(img[None])[0] for img in ref_imgs]
                            for ref_imgs in pixel_values_src]
            cond_latents = cond_latents * 2
        else:
            text_inputs = self.prepare_text2image_prompts(prompt + cfg_prompt)
            cond_latents = None

        hidden_states = self.meta_queries[None].expand(2*b, self.num_queries, -1)
        inputs = self.prepare_forward_input(query_embeds=hidden_states, **text_inputs)

        output = self.llm(**inputs, return_dict=True)

        # hidden_states = output.last_hidden_state[:, -self.num_queries:] #query only
        hidden_states = output.last_hidden_state 
        pooled_out, seq_out = self.llm2dit(hidden_states)

        pipeline = StableDiffusion3Pipeline(
            transformer=self.transformer,
            scheduler=self.test_scheduler,
            vae=self.vae,
            text_encoder=None,
            tokenizer=None,
            text_encoder_2=None,
            tokenizer_2=None,
            text_encoder_3=None,
            tokenizer_3=None,
        )

        pipeline.set_progress_bar_config(disable=not progress_bar)

        samples = pipeline(
            height=height,
            width=width,
            guidance_scale=cfg_scale,
            num_inference_steps=num_steps,
            prompt_embeds=seq_out[:b],
            pooled_prompt_embeds=pooled_out[:b],
            negative_prompt_embeds=seq_out[b:],
            negative_pooled_prompt_embeds=pooled_out[b:],
            generator=generator,
            output_type='latent',
            cond_latents=cond_latents
        ).images.to(self.dtype)

        return self.latents_to_pixels(samples)

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

    def get_sigmas(self, timesteps, n_dim=4):
        sigmas = self.train_scheduler.sigmas.to(device=self.device, dtype=self.dtype)
        schedule_timesteps = self.train_scheduler.timesteps.to(self.device)
        timesteps = timesteps.to(self.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma


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
    from PIL import Image
    import numpy as np


    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='log file path.')
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--prompt", type=str, default='a dog on the left and a cat on the right')
    parser.add_argument("--cfg_prompt", type=str, default='')
    parser.add_argument("--cfg_scale", type=float, default=4.0)
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
