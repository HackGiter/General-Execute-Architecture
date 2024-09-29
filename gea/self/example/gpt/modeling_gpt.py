import math
from typing import Optional, List
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

from transformers import PretrainedConfig, PreTrainedModel
from transformers.utils import ModelOutput
from transformers.activations import ACT2FN

@dataclass
class SequenceOutputs(ModelOutput):
    loss: torch.Tensor = None
    scores: torch.Tensor = None
    logits: torch.FloatTensor = None
    hidden_states: torch.Tensor = None

class GPTConfig(PretrainedConfig):
    model_type = "gpt"

    def __init__(
        self,
        vocab_size: int = None,
        hidden_size: int = None,
        intermediate_size: int = None,
        hidden_act: str = "silu",
        num_heads: int = None,
        num_key_value_heads: int = None,
        num_hidden_layers: int = None,
        rope_theta: float = 10000.0,
        rope_scaling: float = None,
        rms_norm_eps: float = 1e-6,
        padding_idx: int = None,
        initializer_range: float = 0.02,
        tie_word_embeddings: bool = False,
        max_position_embeddings=2048,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_hidden_layers = num_hidden_layers
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.rms_norm_eps = rms_norm_eps
        self.padding_idx = padding_idx
        self.initializer_range = initializer_range
        self.tie_word_embeddings = tie_word_embeddings
        self.max_position_embeddings = max_position_embeddings
        super().__init__(**kwargs)

class GPTPretrainedModel(PreTrainedModel):
    config_class=GPTConfig
    base_model_prefix="model"
    
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

class GPTRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        GPTRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states:torch.Tensor):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class GPTRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # For BC we register cos and sin cached
        self.max_seq_len_cached = max_position_embeddings

    @torch.no_grad()
    def forward(self, x:torch.Tensor, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class GPTAttention(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        
        self.qkv = nn.Linear(self.hidden_size, self.hidden_size * 3, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self._init_rope()

    def _init_rope(self):
        self.rotary_emb = GPTRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(
            self, 
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            ) -> List[torch.Tensor]:
        bsz, seq_len, _ = hidden_states.size()
        
        q, k, v = torch.chunk(self.qkv(hidden_states), chunks=3, dim=-1)
        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(v, position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        k, v = repeat_kv(k, self.num_key_value_groups), repeat_kv(v, self.num_key_value_groups)

        attn_scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        attn_scores = torch.softmax(attn_scores, dim=-1, dtype=torch.float32).to(q.dtype)
        
        hidden_states = torch.matmul(attn_scores, v).transpose(1, 2).contiguous()
        hidden_states = hidden_states.reshape(bsz, seq_len, -1)
        hidden_states = self.o_proj(hidden_states)

        return hidden_states

class GPTMLP(nn.Module):
    def __init__(self, config:GPTConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.up_g_proj = nn.Linear(self.hidden_size, self.intermediate_size * 2, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(
            self,
            hidden_states: torch.Tensor) -> torch.Tensor:
        up_states, gate_states = torch.chunk(self.up_g_proj(hidden_states), chunks=2, dim=-1)
        hidden_states = self.down_proj(self.act_fn(gate_states) * up_states)
        return hidden_states

class GPTDecoder(nn.Module):
    def __init__(self, config:GPTConfig):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = GPTAttention(config=config)

        self.mlp = GPTMLP(config)
        self.input_layernorm = GPTRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GPTRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
            self,
            hidden_states:torch.Tensor = None,
            attention_mask:Optional[torch.Tensor] = None,
            position_ids:Optional[torch.Tensor] = None,) -> torch.Tensor:
        
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
        ) + residual

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states) + residual

        return hidden_states

class GPTModel(GPTPretrainedModel):
    def __init__(self, config: GPTConfig, **kwargs):
        super().__init__(config)
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.padding_idx)
        self.decoders = nn.ModuleList([
            GPTDecoder(config) for _ in range(config.num_hidden_layers)
        ])
        self.norm = GPTRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def cross_entropy(self, logits:torch.Tensor, labels:torch.Tensor) -> torch.Tensor:
        logits = logits[:, :-1].contiguous().view(-1, self.config.vocab_size)
        labels = labels[:, 1:].contiguous().view(-1)
        loss_fct = CrossEntropyLoss()
        return loss_fct(logits, labels)
    
    def update_causal_mask(
            self,
            attention_mask: torch.Tensor,
            input_tensor: torch.Tensor,
            position_ids: torch.Tensor,
    ):
        dtype, device = input_tensor.dtype, input_tensor.device
        seq_len = input_tensor.shape[1]
        causal_mask = torch.full(
            (seq_len, seq_len), fill_value=torch.finfo(dtype).min, dtype=dtype, device=device
        )
        causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(seq_len, device=device) > position_ids.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()
            mask_len = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_len] + attention_mask[:, None, None, :]
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_len] = causal_mask[:, :, :, :mask_len].masked_fill(
                padding_mask, value=torch.finfo(dtype).min
            )
        return causal_mask

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_scores: bool = False,
        output_hidden_states: bool = False,
        **kwargs,
    ) -> SequenceOutputs:
        inputs_embeds = self.embed_tokens(input_ids)

        if position_ids is None:
            position_ids = torch.arange(
                0, inputs_embeds.shape[1], device=inputs_embeds.device
            ).unsqueeze(0)
        if attention_mask is None:
            attention_mask = self.update_causal_mask(
                attention_mask, inputs_embeds, position_ids
            )

        hidden_states = inputs_embeds
        for decoder in self.decoders:
            hidden_states = decoder(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )

        hidden_states = self.norm(hidden_states)
        scores = self.lm_head(hidden_states)
        
        if kwargs.get("logits", None) is None and kwargs.get("labels", None) is not None:
            labels = kwargs.pop("labels", None)
            return SequenceOutputs(
                loss=self.cross_entropy(F.softmax(scores.float(), dim=-1), labels),
                logits=None,
                hidden_states=hidden_states if output_hidden_states else None,
            )
        logits = F.softmax(hidden_states, dim=-1)

        return SequenceOutputs(
            scores=scores if output_scores else None,
            logits=logits,
            hidden_states=hidden_states if output_hidden_states else None,
        )