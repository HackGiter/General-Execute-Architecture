import os
import math
import json
import random
from dataclasses import dataclass
from typing import Union, List, Dict, Any

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import Dataset
import torch.nn.functional as F

from transformers.models.llama.modeling_llama import LlamaRMSNorm

from gea.args import TrainArguments, DataArguments
from gea.utils.callback import TrainState
from gea.execute.executor import run_train
from gea.train.scheduler import get_schedulers
from gea.utils.logging import get_logger
from gea.utils.tools import get_parameter_names
from .modeling_xAdaptive import xAdaptiveConfig, xAdaptiveForCausalLM

logger = get_logger(__name__)

def load_x_model(model:xAdaptiveForCausalLM, pretrained_model_path:str, **kwargs) -> xAdaptiveForCausalLM:
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
        "model.embed_tokens": embed_tokens,
        "lm_head": lm_head,
    })
    model.load_state_dict(model_state_dict)
    frozen_params = kwargs.pop("frozen_params", [])
    for name, param in model.named_parameters():
        logger.debug(f"{name}")
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
class DataCollator4x:
    dtype: torch.dtype = torch.float16
    def __call__(
            self, 
            features: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Dict[str, Any]:
        max_length = max([feature['input_ids'].shape[-1] for feature in features])
        input_ids = torch.cat([
            torch.nn.functional.pad(feature['input_ids'], pad=[0, max_length - feature['input_ids'].shape[-1]], mode='constant', value=0)
            for feature in features
        ], dim=0)
        labels = torch.cat([
            torch.nn.functional.pad(feature['masks'], pad=[0, max_length - feature['input_ids'].shape[-1]], mode='constant', value=0)
            for feature in features
        ], dim=0) > 0
        next_states = torch.cat([
            torch.nn.functional.pad(feature['final_states'], pad=[0, 0, 0, max_length - feature['input_ids'].shape[-1]], mode='constant', value=0)
            for feature in features
        ], dim=0).to(self.dtype)
        hidden_states = torch.cat([
            torch.nn.functional.pad(feature['hidden_states'], pad=[0, 0, 0, max_length - feature['input_ids'].shape[-1]], mode='constant', value=0)
            for feature in features
        ], dim=0).to(self.dtype)
        hidden_states = hidden_states + (torch.rand_like(hidden_states) - 0.5) * 0.25 * 512 / max_length
        return { 
            'input_ids':input_ids, 
            'hidden_states':hidden_states, 
            'next_states':next_states,
            'labels':labels
            }

def get_dataset_fn(
    train_args:TrainArguments, 
    data_args:DataArguments, 
    **kwargs) -> Dict[str, Dataset]:

    with train_args.main_process_first():
        train_dataset_files, eval_datasets_files = [os.path.join(data_args.dataset[0], item) for item in os.listdir(data_args.dataset[0])], None
        if kwargs.get("selective_idx", None) is not None:
            with open(kwargs.pop("selective_idx"), 'r') as f:
                selective_idx = json.load(f)
            _selective_dataset_files = [item for i, item in enumerate(train_dataset_files) if i in selective_idx[kwargs.get("category")]]
            eval_datasets_files = [item for i, item in enumerate(train_dataset_files) if i not in selective_idx[kwargs.get("category")]]
            if train_args.val_ratio > 0:
                eval_samples = math.ceil(len(eval_datasets_files) * train_args.val_ratio)
                eval_datasets_files = random.sample(eval_datasets_files, k=eval_samples)
            train_dataset_files = _selective_dataset_files
        else:
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

def prepare_optimizer_fn(optim_cls:Optimizer, model:xAdaptiveForCausalLM, **kwargs) -> Optimizer:
    general_parameters = get_parameter_names(
        model,
        [LlamaRMSNorm],
    )
    optim_grouped_params = [
        {
            'params': [param for name, param in model.named_parameters() if param.requires_grad and name in general_parameters],
            'lr': kwargs.get('lr', None),
            'weight_decay': kwargs.get('weight_decay', 0.0),
        },
        {
            'params': [param for name, param in model.named_parameters() if param.requires_grad and name not in general_parameters],
            'lr': kwargs.get('lr', None),
            'weight_decay': 0,
        }
    ]
    logger.debug(f"{kwargs}")
    return optim_cls(
        params=optim_grouped_params,
        **kwargs,
    )

def prepare_lr_scheduler_fn(
    name:str,
    optimizer:Optimizer,
    **kwargs,
) -> LRScheduler:
    return get_schedulers(
        name=name,
        optimizer=optimizer,
        **kwargs,
    )

def kl_divergence(
        target_distribution:torch.Tensor, 
        log_predicted_distribution:torch.Tensor, 
        labels:torch.Tensor, 
        temperature=1.0) -> torch.Tensor:
    kl_loss = nn.KLDivLoss(reduction="none")
    divergence = kl_loss(log_predicted_distribution, target_distribution)
    padding_mask = labels > 0
    padding_mask = padding_mask[:, :, None]
    divergence = divergence * padding_mask
    divergence = divergence.sum() / padding_mask.sum()
    return divergence * temperature ** 2

def smooth_l1(
        target:torch.Tensor, 
        predicted:torch.Tensor, 
        labels:torch.Tensor, 
        ) -> torch.Tensor:
    sl1_loss = nn.SmoothL1Loss(reduction="none")
    l1_loss = sl1_loss(predicted, target)
    padding_mask = labels > 0
    padding_mask = padding_mask[:, :, None]
    l1_loss = torch.mean(l1_loss * padding_mask, dim=-1).sum() / padding_mask.sum()
    return l1_loss

@torch.no_grad()
def accuracy(
    target:torch.Tensor, 
    predicted:torch.Tensor, 
    labels:torch.Tensor) -> torch.Tensor:
    target = torch.argmax(target, dim=-1)
    predicted = torch.argmax(predicted, dim=-1)
    padding_mask = labels > 0
    acc_ = (target == predicted) * padding_mask
    acc_ = torch.sum(acc_) / padding_mask.sum()
    return acc_

@torch.no_grad()
def topkacc(
    target:torch.Tensor, 
    predicted:torch.Tensor, 
    labels:torch.Tensor, 
    k:int) -> torch.Tensor:
    target = torch.topk(target, k=k, dim=-1).indices
    predicted = torch.topk(predicted, k=k, dim=-1).indices
    padding_mask = labels > 0
    padding_mask = padding_mask[:, :, None]
    acc_ = (target == predicted) * padding_mask
    acc_ = torch.sum(acc_) / (padding_mask.sum() * k)
    return acc_

def compute_loss(
        model:xAdaptiveForCausalLM, 
        batch:Dict[str, torch.Tensor], 
        state:TrainState, 
        **kwargs) -> Dict[str, Any]:
    next_k = 3
    temperature = kwargs.pop('temperature', 2.0)
    kl_weight = kwargs.pop('kl_weight', 0.5)
    # weights = F.softmax(torch.tensor([11, 8, 5, 3, 1], dtype=torch.float32) / (state.global_epoch + 1), dim=-1)
    weights = F.softmax(torch.tensor([11, 8, 5], dtype=torch.float32) / (state.global_epoch + 1), dim=-1)

    batch = { k:v.to(model.device) for k, v in batch.items() }
    outputs = model(
            input_ids=batch['input_ids'],
            hidden_states=batch['next_states'][:, :-1],
            prev_hidden_states=batch['hidden_states'][:, :-1],
            labels=batch['next_states'],
            output_hidden_states=False,
            return_dict=True,
            use_cache=False,
        )
    with torch.no_grad():
        target_logits = outputs.target_logits
        target_distributions = F.softmax(target_logits / temperature, dim=-1)

    avg_loss, avg_acc = 0, 0
    losses, accs = [None] * next_k, [None] * next_k
    logits = torch.split(outputs.logits, 1, 0)
    for i, (logit, weight) in enumerate(zip(logits, weights)):
        hidden_state = outputs.hidden_states[i][None, ...] if i == 0 else outputs.hidden_states[i, i:][None, ...]
        distributions = F.softmax((logit if i == 0 else logit[:, :-i]) / temperature, dim=-1)
        log_distribution = F.log_softmax((logit if i == 0 else logit[:, :-i]) / temperature, dim=-1)
        loss = kl_divergence(target_distributions[:, 1 + i:], log_distribution, batch["labels"][:, 1 + i:], temperature) * kl_weight + \
            smooth_l1(batch['next_states'][:, 1 + i:], hidden_state, batch["labels"][:, 1 + i:]) * (1 - kl_weight)
        acc = accuracy(target_distributions[:, 1 + i:], distributions, batch['labels'][:, 1 + i:])
        losses[i] = loss.detach()
        accs[i] = acc.detach()
        avg_loss += loss * weight
        avg_acc += acc * weight
    metrics = {"accuracy" : avg_acc, "losses": losses, "accs": accs }
    return avg_loss, metrics

@torch.no_grad()
def execute_metrics(
    model:xAdaptiveForCausalLM, 
    batch:Dict[str, torch.Tensor], 
    state:TrainState,
    **kwargs) -> Dict[str, Any]:
    model.train()
    next_k = 3
    kl_weight = kwargs.pop('kl_weight', 0.5)
    # weights = F.softmax(torch.tensor([11, 8, 5, 3, 1], dtype=torch.float32) / (state.global_epoch + 1), dim=-1)
    weights = F.softmax(torch.tensor([11, 8, 5], dtype=torch.float32) / (state.global_epoch + 1), dim=-1)

    batch = { k:v.to(model.device) for k, v in batch.items() }
    outputs = model(
            input_ids=batch['input_ids'],
            hidden_states=batch['next_states'][:, :-1],
            prev_hidden_states=batch['hidden_states'][:, :-1],
            labels=batch['next_states'],
            output_hidden_states=False,
            return_dict=True,
            use_cache=False,
        )
    target_logits = outputs.target_logits
    target_distributions = F.softmax(target_logits, dim=-1)
    
    avg_loss, avg_acc, avg_topk = 0, 0, 0
    losses, accs, topks = [None] * next_k, [None] * next_k, [None] * next_k
    logits = torch.split(outputs.logits, 1, 0)
    for i, (logit, weight) in enumerate(zip(logits, weights)):
        hidden_state = outputs.hidden_states[i][None, ...] if i == 0 else outputs.hidden_states[i, i:][None, ...]
        distributions = F.softmax(logit if i == 0 else logit[:, :-i], dim=-1)
        log_distribution = F.log_softmax(logit if i == 0 else logit[:, :-i], dim=-1)
        loss = kl_divergence(target_distributions[:, 1 + i:], log_distribution, batch["labels"][:, 1 + i:], 1.0) * kl_weight + \
            smooth_l1(batch['next_states'][:, 1 + i:], hidden_state, batch["labels"][:, 1 + i:]) * (1 - kl_weight)
        acc = accuracy(target_distributions[:, 1 + i:], distributions, batch['labels'][:, 1 + i:])
        topk = topkacc(target_distributions[:, 1 + i:],  distributions, batch['labels'][:, 1 + i:], 5)
        losses[i] = loss.detach()
        accs[i] = acc.detach()
        topks[i] = topk.detach()
        avg_loss += loss * weight
        avg_acc += acc * weight
        avg_topk += topk * weight
    metrics = { "accuracy":avg_acc, "topk":avg_topk, "losses":losses, "accs":accs, "topks":topks }
    return avg_loss, metrics

def main():
    kwargs = {
        "load_model_fn": load_x_model,
        "get_dataset_fn": get_dataset_fn,
        "dataset_num_proc": 8,
        "batched": False,
        "train_collate_fn": DataCollator4x(dtype=torch.bfloat16),
        "eval_collate_fn": DataCollator4x(dtype=torch.bfloat16),
        "compute_loss": compute_loss,
        "execute_metrics": execute_metrics,
        "prepare_optimizer_fn": prepare_optimizer_fn,
        "prepare_lr_scheduler_fn": prepare_lr_scheduler_fn,
        "lr_scheduler_kwargs": {
            "min_lr_rate": 0.1,
        },
        "inputs_kwargs": {
            "temperature": 2.0,
            "kl_weight": 0.5,
        },
        "model_kwargs": {
            "config_cls": xAdaptiveConfig,
            "model_cls": xAdaptiveForCausalLM,
            "pretrained_model_path": "/data/lhz/models/meta/Llama-2-7b-chat-hf",
            "frozen_params": ['model.embed_tokens.weight', 'lm_head.weight'],
        }, 
        "data_kwargs": {
            # "category": "up_sl_entp_tks",
            "category": "dn_sl_entp_tks",
            # "category": "up_sl_ppl_tks",
            # "selective_idx": "/home/lhz/Workplace/General Execute Architecture/gea/self/xAdaptive/analysis/datas/selective_ppls_entropy_v1.json",
            "selective_idx": "/home/lhz/Workplace/General Execute Architecture/gea/self/xAdaptive/analysis/datas/selective_ppls_entropy_v2.json",
        },    
    }
    run_train(kwargs)

if __name__ == "__main__":
    main()