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

class N2MGramConfig(PretrainedConfig):
    model_type = "n2mgram"

    def __init__(
        self,
        n: int = None,
        m: int = None,
        vocab_size: int = None,
        hidden_size: int = None,
        intermediate_size: int = None,
        hidden_act: str = None,
        padding_idx: int = None,
        initializer_range: float = 0.02,
        tie_word_embeddings: bool = False,
        **kwargs
    ):
        self.n = n
        self.m = m
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.padding_idx = padding_idx
        self.initializer_range = initializer_range
        self.tie_word_embeddings = tie_word_embeddings
        super().__init__(**kwargs)

class N2MGramPretrainedModel(PreTrainedModel):
    config_class=N2MGramConfig
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

class N2MGramProject(nn.Module):
    def __init__(self, config: N2MGramConfig) -> None:
        super().__init__()
        self.n = config.n
        self.m = config.m
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.act_fn = ACT2FN[config.hidden_act]
        self.projs = nn.Sequential(
            nn.Linear(self.hidden_size * self.n, self.intermediate_size),
            self.act_fn,
            nn.Linear(self.intermediate_size, self.hidden_size * self.m),
        ) if self.intermediate_size is not None else nn.Sequential(
            nn.Linear(self.hidden_size * self.n, self.hidden_size * self.m)
        )

    def forward(self, hidden_states:torch.Tensor) -> torch.Tensor:
        return self.projs(hidden_states)

class N2MGramModel(N2MGramPretrainedModel):
    def __init__(self, config: N2MGramConfig, **kwargs):
        super().__init__(config)
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.padding_idx)
        self.proj = N2MGramProject(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def compute_loss(self, logits:torch.Tensor, labels:torch.Tensor) -> torch.Tensor:
        logits = logits.contiguous().view(-1, self.config.vocab_size)
        labels = labels.contiguous().view(-1)
        loss_fct = CrossEntropyLoss()
        return loss_fct(logits, labels)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        output_hidden_states: bool = False,
        **kwargs,
    ) -> SequenceOutputs:
        bsz, _ = input_ids.size()
        hidden_states = self.embed_tokens(input_ids)
        hidden_states = self.proj(hidden_states.view(bsz, -1))
        hidden_states = self.lm_head(hidden_states.view(bsz, -1, self.config.hidden_size))
        logits = F.softmax(hidden_states, dim=-1)
        if kwargs.get("labels", None) is not None:
            labels = kwargs.pop("labels")
            if labels is not None:
                return SequenceOutputs(
                    loss=self.compute_loss(logits, labels),
                    logits=None,
                    hidden_states=hidden_states if output_hidden_states else None,
                )
    
        return SequenceOutputs(
            logits=logits,
            hidden_states=hidden_states if output_hidden_states else None,
        )