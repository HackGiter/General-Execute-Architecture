import os
from typing import List, Dict, Any

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers import get_scheduler

from gea.execute.executor import run_train
from gea.utils.logging import get_logger
from .modeling_n2m import N2MGramConfig, N2MGramModel

logger = get_logger(__name__)

def load_n2m_model(model:N2MGramModel, pretrained_model_path:str, **kwargs) -> N2MGramModel:
    embed_tokens, lm_head = None, None
    state_dict_files = sorted([item for item in os.listdir(pretrained_model_path) if item.endswith('.bin')])
    pretrained_state_dict = torch.load(os.path.join(pretrained_model_path, state_dict_files[0]), map_location='cpu')
    for key, value in pretrained_state_dict.items():
        if "embed_tokens" in key:
            embed_tokens = value
            logger.info("Pretrained embed tokens loaded")
    pretrained_state_dict = torch.load(os.path.join(pretrained_model_path, state_dict_files[-1]), map_location='cpu')
    for key, value in pretrained_state_dict.items():
        if "lm_head" in key:
            logger.info("Pretrained lm head loaded")
            lm_head = value
    model_state_dict = model.state_dict()
    model_state_dict.update({
        "embed_tokens": embed_tokens,
        "lm_head": lm_head,
    })
    model.load_state_dict(model_state_dict)
    frozen_params = kwargs.pop("frozen_params", [])
    for name, param in model.named_parameters():
        if name in frozen_params:
            param.requires_grad = False
    return model      

def postprocess_dataset(examples:List[Dict[str, Any]], n:int, m:int, **kwargs) -> Dict[str, List[Any]]:
    k = n + m
    outputs = { "input_ids" : [], "labels" : [] }
    for item in examples['input_ids']:
        item = [item[0][i:i + k] for i in range(len(item[0]) - k)]
        outputs["input_ids"].extend([it[:n] for it in item])
        outputs["labels"].extend([it[-m:] for it in item])
    return outputs

def prepare_optimizer_fn(optim_cls:Optimizer, model:N2MGramModel, **kwargs) -> Optimizer:
    optim_grouped_params = [
        {
            'params': [param for _, param in model.named_parameters() if param.requires_grad],
            'lr': kwargs.pop('lr', None),
            'weight_decay': kwargs.pop('weight_decay', 0.0),
        },
    ]
    return optim_cls(
        params=optim_grouped_params,
        **kwargs,
    )

def prepare_lr_scheduler_fn(
    name:str,
    optimizer:Optimizer,
    **kwargs,
) -> LRScheduler:
    return get_scheduler(
        name=name,
        optimizer=optimizer,
        **kwargs,
    )

def main():
    kwargs = {
        "config_cls": N2MGramConfig,
        "model_cls": N2MGramModel,
        "load_model_fn": load_n2m_model,
        "pretrained_model_path": "/data/lhz/models/meta/Llama-2-7b-chat-hf",
        "frozen_params": ['embed_tokens.weight', 'lm_head.weight'],
        "dataset_num_proc": 16,
        "batched": False,
        "postprocess_dataset": postprocess_dataset,
        "postprocess_remove_columns": ["attention_mask"],
        "prepare_optimizer_fn": prepare_optimizer_fn,
        "prepare_lr_scheduler_fn": prepare_lr_scheduler_fn,
        "postprocess_kwargs": {
            "n": 2,
            "m": 4,
            "batched": True,
        },
        "lr_scheduler_kwargs": {
            "min_lr_rate": 0.1,
        },        
    }
    run_train(kwargs)

if __name__ == "__main__":
    main()