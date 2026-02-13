import torch
import torch.nn as nn
from transformers import Qwen2PreTrainedModel, Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm, Qwen2DecoderLayer


class Qwen2Connector(Qwen2PreTrainedModel):
    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        for layer in self.layers:
            layer.self_attn.is_causal = False

        self._attn_implementation = config._attn_implementation
        assert self._attn_implementation == 'flash_attention_2'
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, inputs_embeds):
        position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
        position_ids = position_ids.expand(inputs_embeds.shape[0], -1)
        hidden_states = inputs_embeds

        for encoder_layer in self.layers:
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    None,
                    position_ids,
                    use_reentrant=False
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask=None,
                    position_ids=position_ids,
                )

            hidden_states = layer_outputs[0]

        hidden_states = self.norm(hidden_states)

        return hidden_states
