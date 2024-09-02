from typing import List
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
    logits: torch.FloatTensor = None
    hidden_states: torch.Tensor = None

class xHeadConfig(PretrainedConfig):
    model_type = "xhead"

    def __init__(
        self,
        k: int = None,
        ratio: float = None,
        vocab_size: int = None,
        hidden_size: int = None,
        output_size: int = None,
        intermediate_size: int = None,
        hidden_act: str = None,
        padding_idx: int = None,
        initializer_range: float = 0.02,
        tie_word_embeddings: bool = False,
        **kwargs
    ):
        self.k = k
        self.ratio = ratio
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.padding_idx = padding_idx
        self.initializer_range = initializer_range
        self.tie_word_embeddings = tie_word_embeddings
        super().__init__(**kwargs)

class xHeadPretrainedModel(PreTrainedModel):
    config_class=xHeadConfig
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

class xHeadProject(nn.Module):
    def __init__(self, config: xHeadConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.output_size = config.output_size
        self.intermediate_size = config.intermediate_size
        self.k = config.k
        self.ratio = config.ratio
        self.g_proj = nn.Linear(self.hidden_size + self.intermediate_size, 
                                self.hidden_size + self.intermediate_size, 
                                bias=False)
        self.u_proj = nn.Linear(self.hidden_size * 2, self.output_size, bias=False)
        self.o_proj = nn.Linear(self.output_size, self.hidden_size, bias=False)
        self.u_act_fn = nn.GELU()
        self.g_act_fn = nn.Tanh()

    def forward(self, 
                inputs_embeds:torch.Tensor,
                hidden_states:torch.Tensor) -> List[torch.Tensor]:
        output_states = []
        for i in range(self.k):
            sub_input_embeds = inputs_embeds if i == 0 else inputs_embeds[:, i:]
            sub_hidden_state = hidden_states if i == 0 else hidden_states[:, i:]
            pre_hidden_state = None if i == 0 else pre_hidden_state[:, 1:]

            u_proj_states = self.u_act_fn(
                self.u_proj(
                    torch.cat(
                        [sub_hidden_state, sub_input_embeds]
                        , dim=-1)))
            g_proj_states = self.g_proj(
                torch.cat([pre_hidden_state, sub_input_embeds], dim=-1)) if pre_hidden_state is not None \
                    else F.linear(sub_input_embeds,
                                   self.g_proj.weight[:, -self.hidden_size:])
            
            u_proj_states = self.o_proj(
                u_proj_states * g_proj_states[..., :self.output_size]) + sub_hidden_state
            output_states.append(u_proj_states)
            
            pre_hidden_state =  pre_hidden_state * self.ratio + \
                self.g_act_fn(g_proj_states[..., self.output_size:]) * (1 - self.ratio) \
                    if pre_hidden_state is not None \
                        else self.g_act_fn(g_proj_states[..., self.output_size:])
        return output_states

class xHeadModel(xHeadPretrainedModel):
    def __init__(self, config: xHeadConfig, **kwargs):
        super().__init__(config)
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.padding_idx)
        self.proj = xHeadProject(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def cross_entropy(self, logits:List[torch.Tensor], labels:torch.Tensor) -> torch.Tensor:
        loss, loss_fct = 0, CrossEntropyLoss()
        for i, logit in enumerate(logits):
            loss += loss_fct(logit[:, i:].view(-1, self.config.vocab_size), labels[:, i:].view(-1)) if i != 0 \
                else loss_fct(logit.view(-1, self.config.vocab_size), labels.view(-1))
        return loss
    
    def kl_loss(self, logits:torch.Tensor, labels:torch.Tensor, masks:torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        loss, masks, kl_fct = 0, masks[..., None], nn.KLDivLoss(reduction="none")
        for i, logit in enumerate(logits):
            if i != 0:
                loss += (kl_fct(torch.log_softmax(logit, -1), labels[:, i:]) * masks[:, i:]).sum() / masks[:, i:].sum()
            else:
                loss += (kl_fct(torch.log_softmax(logit, -1), labels) * masks).sum() / masks.sum()
        return loss * temperature ** 2

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        hidden_states: torch.Tensor = None,
        output_hidden_states: bool = False,
        **kwargs,
    ) -> SequenceOutputs:
        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = self.proj(inputs_embeds, hidden_states)
        logits = [self.lm_head(sub_hidden_state).float() for sub_hidden_state in hidden_states]
        if kwargs.get("logits", None) is None:
            labels = kwargs.pop("labels")
            return SequenceOutputs(
                loss=self.cross_entropy([F.softmax(logit, dim=-1) for logit in logits], labels),
                logits=None,
                hidden_states=hidden_states if output_hidden_states else None,
            )
        elif kwargs.get("logits", None) is not None:
            temperature = kwargs.pop("temperature", 1.0)
            labels = torch.softmax(self.lm_head(kwargs.pop("logits")).float() / temperature, dim=-1)
            masks = kwargs.pop("masks")
            logits = [logit / temperature for logit in logits]
            return SequenceOutputs(
                loss=self.kl_loss(logits, labels, masks, temperature),
                logits=None,
                hidden_states=hidden_states if output_hidden_states else None
            )
    
        return SequenceOutputs(
            logits=logits,
            hidden_states=hidden_states if output_hidden_states else None,
        )