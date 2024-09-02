from typing import (
    Optional,
    List,
    Tuple,
    Union
)

import math
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

from transformers import (
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.models.llama.modeling_llama import (
    LlamaRMSNorm,
    AttentionMaskConverter,
    LlamaRotaryEmbedding,
    LlamaLinearScalingRotaryEmbedding,
    LlamaDynamicNTKScalingRotaryEmbedding,
    repeat_kv
)
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
from transformers.utils import logging, replace_return_docstrings
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(
        q:torch.Tensor, 
        k:torch.Tensor, 
        cos:torch.Tensor, 
        sin:torch.Tensor, 
        training:bool=False):
    q_len, k_len = q.shape[-2], k.shape[-2]
    cos, sin = cos[:, None, None, ...], sin[:, None, None, ...]
    q_embed = (q * cos[..., -q_len:, :]) + (rotate_half(q) * sin[..., -q_len:, :])
    k_embed = (k * cos[..., :k_len, :]) + (rotate_half(k) * sin[..., :k_len, :])
    return q_embed, k_embed


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "AdaptiveConfig"

class xAdaptiveConfig(PretrainedConfig):
    model_type = "xadaptive"
    def __init__(
            self,
            k=5,
            vocab_size=32000,
            hidden_size=4096,
            ouptut_size=4096,
            intermediate_size=11008,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=None,
            hidden_act="silu",
            max_position_embeddings=2048,
            initializer_range=0.02,
            rms_norm_eps=1e-6,
            use_cache=True,
            pad_token_id=None,
            bos_token_id=1,
            eos_token_id=2,
            pretraining_tp=1,
            tie_word_embeddings=False,
            rope_theta=10000.0,
            rope_scaling=None,
            attention_bias=False,
            attention_dropout=0.0,
            **kwargs,
            ):
        self.k = k
        self.vocab_size=vocab_size
        self.ouptut_size = ouptut_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range=initializer_range
        self.rms_norm_eps=rms_norm_eps
        self.use_cache=use_cache
        self.pad_token_id=pad_token_id
        self.bos_token_id=bos_token_id
        self.eos_token_id=eos_token_id
        self.pretraining_tp=pretraining_tp
        self.tie_word_embeddings=tie_word_embeddings
        self.rope_theta=rope_theta
        self.rope_scaling=rope_scaling
        self.attention_bias=attention_bias
        self.attention_dropout=attention_dropout
        super().__init__(            
            vocab_size,
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            hidden_act,
            max_position_embeddings,
            initializer_range,
            rms_norm_eps,
            use_cache,
            pad_token_id,
            bos_token_id,
            eos_token_id,
            pretraining_tp,
            tie_word_embeddings,
            rope_theta,
            rope_scaling,
            attention_bias,
            attention_dropout,
            **kwargs,
            )

class xAdaptiveMLP(nn.Module):
    def __init__(self, config:xAdaptiveConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj

class xAdaptiveAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: xAdaptiveConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.k = config.k

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim * self.k, bias=False)
        self.k_proj = nn.Linear(self.hidden_size * 2, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size * 2, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim, max_position_embeddings=self.max_position_embeddings, scaling_factor=scaling_factor
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim, max_position_embeddings=self.max_position_embeddings, scaling_factor=scaling_factor
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        prev_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        q_bsz, q_len, _ = hidden_states.size()
        kv_bsz, kv_seq_len, _ = prev_hidden_states.size()

        hidden_states = self.input_layernorm(hidden_states)
        prev_hidden_states = torch.cat([
            prev_hidden_states, hidden_states[:, :-1]
        ], dim=-1)

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(prev_hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(prev_hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)
        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(prev_hidden_states)
            value_states = self.v_proj(prev_hidden_states)

        query_states = query_states.view(q_bsz, self.k, -1, self.num_heads, self.head_dim).transpose(2, 3)
        key_states = key_states.view(kv_bsz, 1, -1, self.num_key_value_heads, self.head_dim).transpose(2, 3)
        value_states = value_states.view(kv_bsz, 1, -1, self.num_key_value_heads, self.head_dim).transpose(2, 3)

        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(
            q=query_states, 
            k=key_states, 
            cos=cos, 
            sin=sin, 
            training=self.training)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(3, 4)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (q_bsz, self.k, self.num_heads, q_len, kv_seq_len) and not self.training:
            raise ValueError(
                f"Attention weights should be of size {(q_bsz, self.k, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (q_bsz, 1, 1, q_len, kv_seq_len) and not self.training:
                raise ValueError(
                    f"Attention mask should be of size {(q_bsz, 1, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

    
        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (q_bsz, self.k, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(q_bsz, self.k, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.permute(1, 0, 3, 2, 4).contiguous()
        attn_output = attn_output.reshape(q_bsz * self.k, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

class xAdaptiveDecoderLayer(nn.Module):
    def __init__(self, config: xAdaptiveConfig, id: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.id = id
        self.self_attn = xAdaptiveAttention(config=config, k=config.k)
        self.mlp = xAdaptiveMLP(config)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        prev_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        dummy: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states[:, 1:] if self.training else hidden_states[:, -1:]

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            prev_hidden_states=prev_hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states

        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

class xAdaptivePretrainedModel(PreTrainedModel):
    config_class = xAdaptiveConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["AdaptiveDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

class xAdaptiveHead(nn.Module):
    def __init__(self, config: xAdaptiveConfig, oup_hid_sz:int=4096, intm_hid_sz:int=5120, ratio:float = 0.25) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.output_size = oup_hid_sz
        self.intermediate_size = intm_hid_sz
        self.ratio = ratio
        # self.norm = LlamaRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.g_proj = nn.Linear(self.hidden_size + self.intermediate_size, self.output_size + self.intermediate_size, bias=False)
        self.u_proj = nn.Linear(self.hidden_size * 2, self.output_size, bias=False)
        self.o_proj = nn.Linear(self.output_size, self.hidden_size, bias=False)
        self.u_act_fn = nn.GELU()
        self.g_act_fn = nn.Tanh()

    def forward(
            self, 
            hidden_states: torch.Tensor,
            inputs_embeds: torch.Tensor,
            prev_hidden_states: torch.Tensor,
            ):
        u_proj_states = self.u_act_fn(self.u_proj(torch.cat([hidden_states, inputs_embeds], dim=-1)))
        g_proj_states = self.g_proj(torch.cat([prev_hidden_states, inputs_embeds], dim=-1)) if prev_hidden_states is not None else F.linear(inputs_embeds, self.g_proj.weight[:, -self.hidden_size:])
        u_proj_states = self.o_proj(u_proj_states * g_proj_states[..., :self.output_size])
        prev_hidden_states =  prev_hidden_states * self.ratio + self.g_act_fn(g_proj_states[..., self.output_size:]) * (1 - self.ratio) if prev_hidden_states is not None else self.g_act_fn(g_proj_states[..., self.output_size:])
        return u_proj_states + hidden_states, prev_hidden_states
    
class xAdaptiveModel(xAdaptivePretrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`AdaptiveDecoderLayer`]

    Args:
        config: AdaptiveModel
    """

    def __init__(self, config: xAdaptiveConfig, inp_hid_sz: int = 4096, oup_hid_sz: int = 4096, intm_hid_sz: int = 5120):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.k = config.k

        self.medusa = xAdaptiveHead(config, inp_hid_sz, oup_hid_sz, intm_hid_sz)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        # self.embed_tokens = torch.tensor(0.0, requires_grad=True)
        self.layers = nn.ModuleList([xAdaptiveDecoderLayer(config, id, self.k) for id in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self._tie_or_clone_weights(self.embed_tokens, value)
        for param in self.embed_tokens.parameters():
            param.requires_grad = False

    def _prepare_4d_local_attention_mask(self, input_shape, past_key_values_length, local_attn_size, ahead_attn_size):
        bsz, tgt_len = input_shape
        mask = torch.ones([bsz, 1, tgt_len + past_key_values_length, tgt_len + past_key_values_length], dtype=torch.bool)
        mask = torch.tril(~torch.tril(mask, diagonal=-local_attn_size))
        mask[..., :ahead_attn_size] = 1
        return mask[..., -tgt_len:, :]

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = AttentionMaskConverter._make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _prepare_4d_attention_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )


        return combined_attention_mask
    
    def _prepare_decoder_local_attention_mask(self, attention_mask, inputs_embeds):
        expanded_attn_mask = torch.zeros_like(attention_mask, dtype=inputs_embeds.dtype)
        expanded_attn_mask = expanded_attn_mask.masked_fill(
            attention_mask == 0, torch.finfo(inputs_embeds.dtype).min
        )
        return expanded_attn_mask

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        prev_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        adaptive_states: Optional[torch.Tensor] = None,
        continued: Optional[bool] = False,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if not continued:
            if position_ids is None:
                device = input_ids.device if input_ids is not None else inputs_embeds.device
                if attention_mask is not None:
                    position_ids = attention_mask.cumsum(dim=-1, dtype=torch.long) - 1
                    position_ids.masked_fill_(attention_mask == 0, 1)
                    position_ids = position_ids[:, -seq_length:]
                else:
                    position_ids = torch.arange(
                        past_key_values_length, seq_length_with_past, dtype=torch.long, device=device
                    )
                position_ids = position_ids.unsqueeze(0)
            else:
                position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if not continued:
            if attention_mask is None:
                attention_mask = torch.ones(
                    (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
                )
            if not self.training:
                attention_mask = self._prepare_decoder_attention_mask(
                    attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
                )[:, None, :, -1:, :-1]
            else:
                attention_mask = self._prepare_decoder_attention_mask(
                    attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
                )[:, None, :, 1:, :-1]
        
        hidden_states = inputs_embeds if not continued else prev_hidden_states

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        if not continued:
            for idx, decoder_layer in enumerate(self.layers):
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)

                past_key_value = past_key_values[idx] if past_key_values is not None else None

                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        decoder_layer.__call__,
                        hidden_states,
                        prev_hidden_states,
                        attention_mask,
                        position_ids,
                        self.embed_tokens
                        )
                else:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        prev_hidden_states=prev_hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_value,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                    )

                hidden_states = layer_outputs[0]

                if use_cache:
                    next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

                if output_attentions:
                    all_self_attns += (layer_outputs[1],)
        elif use_cache:
            next_decoder_cache += past_key_values

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.training:

            next_hidden_states, next_hidden_state = [], None
            hidden_states = hidden_states.view(-1, self.k, hidden_states.shape[1], hidden_states.shape[-1])

            output_state, next_hidden_state = self.medusa(
                hidden_states[:, 0], inputs_embeds[:, 1:], next_hidden_state
            )
            next_hidden_states.append(self.norm(output_state))
            for idx in range(1, self.k):
                if self.gradient_checkpointing:
                    output_state, next_hidden_state = self._gradient_checkpointing_func(
                        self.medusa.__call__, hidden_states[:, idx, :-idx], inputs_embeds[:, idx + 1:], next_hidden_state[:, :-1], use_reentrant=False
                        )
                else:
                    output_state, next_hidden_state = self.medusa(
                        hidden_states[:, idx, :-idx], inputs_embeds[:, idx + 1:], next_hidden_state[:, :-1]
                    )
                next_hidden_states.append(self.lm_head(self.norm(output_state)).float())
            hidden_states = next_hidden_states
        else:

            if not continued:
                output_state, next_hidden_state = self.medusa(hidden_states[:, -self.k], inputs_embeds[:, -1], None)
                if output_hidden_states:
                    all_hidden_states += (next_hidden_state, )
                hidden_states = self.norm(output_state)
            else:
                output_state, next_hidden_state = self.medusa(hidden_states, inputs_embeds[:, -1], adaptive_states)
                if output_hidden_states:
                    all_hidden_states += (next_hidden_state, )
                hidden_states = self.lm_head(self.norm(output_state)).float()

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

class xAdaptiveForCausalLM(xAdaptivePretrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.model = xAdaptiveModel(config, **kwargs)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value=value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self._tie_or_clone_weights(self.lm_head, new_embeddings)
        for param in self.lm_head.parameters():
            param.requires_grad = False

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        continued: Optional[bool] = False,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            continued=continued,
            **kwargs,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            if self.training:
                logits = [self.lm_head(hidden_states[0])] + hidden_states[1:]
            else:
                logits = self.lm_head(hidden_states).float() if not continued else hidden_states

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids_length = input_ids.shape[-1] - past_key_values[0][0].shape[-2]
            input_ids = input_ids[:, -input_ids_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids_length:].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past


