import os
import math
import random
from dataclasses import dataclass
from typing import Union, List, Dict, Any

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import Dataset
from transformers import get_scheduler

from gea.args import TrainArguments, DataArguments
from gea.execute.executor import run_train
from gea.utils.logging import get_logger
from .modeling_xhead import xHeadConfig, xHeadModel

logger = get_logger(__name__)

def load_xhead_model(model:xHeadModel, pretrained_model_path:str, **kwargs) -> xHeadModel:
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

class CustomDataset(Dataset):
    def __init__(self, files:List[str]) -> None:
        super().__init__()
        self.files = files

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        return torch.load(self.files[index])
    
    def __getitems__(self, indexs:List[int]) -> List[Dict[str, Any]]:
        return [torch.load(self.files[i]) for i in indexs]

@dataclass
class DataCollatorwithxHead:
    hidden_size: int = 4096
    k: int = 4
    dtype: torch.dtype = torch.float16
    def __call__(
            self, 
            features: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Dict[str, Any]:
        max_length = max([feature['input_ids'].shape[-1] for feature in features])
        input_ids = torch.cat([
            torch.nn.functional.pad(feature['input_ids'], pad=[0, max_length - feature['input_ids'].shape[-1]], mode='constant', value=0)
            for feature in features
        ], dim=0)
        masks = torch.cat([
            torch.nn.functional.pad(feature['masks'], pad=[0, max_length - feature['input_ids'].shape[-1]], mode='constant', value=-100)
            for feature in features
        ], dim=0) > 0
        logits = torch.cat([
            torch.nn.functional.pad(feature['final_states'], pad=[0, 0, 0, max_length - feature['input_ids'].shape[-1]], mode='constant', value=0)
            for feature in features
        ], dim=0).to(self.dtype)
        hidden_states = torch.cat([
            torch.nn.functional.pad(feature['hidden_states'], pad=[0, 0, 0, max_length - feature['input_ids'].shape[-1]], mode='constant', value=0)
            for feature in features
        ], dim=0).to(self.dtype)
        # hidden_states = hidden_states + (torch.rand_like(hidden_states) - 0.5) * 0.25 * 512 / max_length
        return { 
            'input_ids':input_ids, 
            'hidden_states':hidden_states, 
            'logits':logits,
            'masks':masks}

def get_dataset_fn(
    train_args:TrainArguments, 
    data_args:DataArguments, 
    **kwargs) -> Dict[str, Dataset]:

    with train_args.main_process_first():
        train_dataset_files, eval_datasets_files = [os.path.join(data_args.dataset[0], item) for item in os.listdir(data_args.dataset[0])], None
        if train_args.val_ratio > 0:
            eval_samples = math.ceil(len(train_dataset_files) * train_args.val_ratio)
            eval_datasets_files = random.sample(train_dataset_files, k=eval_samples)
            for item in eval_datasets_files:
                train_dataset_files.remove(item)
        train_datasets = CustomDataset(train_dataset_files)
        eval_datasets = CustomDataset(eval_datasets_files) if eval_datasets_files is not None else None

    return {
        'train_dataset': train_datasets,
        'eval_dataset': eval_datasets
    }

def prepare_optimizer_fn(optim_cls:Optimizer, model:xHeadModel, **kwargs) -> Optimizer:
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
        "config_cls": xHeadConfig,
        "model_cls": xHeadModel,
        "load_model_fn": load_xhead_model,
        "get_dataset_fn": get_dataset_fn,
        "pretrained_model_path": "/data/lhz/models/meta/Llama-2-7b-chat-hf",
        "frozen_params": ['embed_tokens.weight', 'lm_head.weight'],
        "dataset_num_proc": 16,
        "batched": False,
        "train_collate_fn": DataCollatorwithxHead(hidden_size=4096, k=4),
        "eval_collate_fn": DataCollatorwithxHead(hidden_size=4096, k=4),
        "prepare_optimizer_fn": prepare_optimizer_fn,
        "prepare_lr_scheduler_fn": prepare_lr_scheduler_fn,
        "lr_scheduler_kwargs": {
            "min_lr_rate": 0.1,
        },
        "inputs_kwargs": {
            "temperature": 2.0,
        },        
    }
    run_train(kwargs)

if __name__ == "__main__":
    main()