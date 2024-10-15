from typing import Dict

import torch
from torch.optim import Optimizer

from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

from .callback import Optim

IGNORE_INDEX=-100

OPTIMIZERS:Dict[str, Optimizer] = {
    Optim.ADAGRAD: torch.optim.Adagrad,
    Optim.ADAMW: torch.optim.AdamW,
    Optim.SGD: torch.optim.SGD,
}

LAYERNORM_NAMES = {"norm", "ln"}