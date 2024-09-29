from functools import partial
from dataclasses import dataclass
from typing import Union, Tuple, List, Dict, Any

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import Dataset
from transformers import get_scheduler
from datasets import load_dataset

from gea.args import TrainArguments, DataArguments, ModelArguments, EvalArguments
from gea.execute.executor import run_train
from gea.utils.logging import get_logger
from .modeling_gpt import GPTConfig, GPTModel

logger = get_logger(__name__)

class GPTTokenizer:
    def __init__(self) -> None:
        self.eos_token_id = 0
        self.eos_token = "<s>"
        self.lookup_tables = {
            self.eos_token: self.eos_token_id,
            '0': 1,
            '1': 2,
            '2': 3,
            '3': 4,
            '4': 5,
            '5': 6,
            '6': 7,
            '7': 8,
            '8': 9,
            '9': 10,
            '.': 11, 
            '=': 12,
            '+': 13,
            '-': 14,
            '*': 15,
            '/': 16,
        }

    def __call__(self, strings: Union[str, List[str]], add_special_tokens:bool = True, **kwargs) -> torch.Tensor:
        if not isinstance(strings, List):
            strings = [strings]
        ids = []
        for item in strings:
            ids.append([self.lookup_tables[s] for s in item] + ([self.eos_token_id] if add_special_tokens else []) )
        return torch.tensor(ids, dtype=torch.long)
    
    def batch_decode(self, ids: Union[torch.Tensor, List[List[int]]]) -> List[str]:
        if isinstance(ids, torch.Tensor):
            ids = ids.cpu().tolist()
        strings = []
        for item in ids:
            tmp = ''
            for i in item:
                tmp += self.lookup_tables[i]
            strings.append(tmp)
        return strings

def get_model_fn(
        model_args:ModelArguments, 
        **kwargs) -> Tuple[GPTModel, GPTTokenizer]:
    config = GPTConfig.from_pretrained(model_args.path)
    model = GPTModel(config)
    tokenizer = GPTTokenizer()
    return model, tokenizer

@dataclass
class DataCollatorwithGPT:
    eos_token_id: int = 0
    def __call__(
            self, 
            features: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Dict[str, Any]:
        seq_length = [feature['input_ids'].shape[-1] + feature['labels'].shape[-1] for feature in features]
        max_length = max(seq_length) + 1
        input_ids = torch.cat([
            torch.nn.functional.pad(
                torch.cat([feature['input_ids'], feature['labels']], dim=-1), 
                pad=[0, max_length - seq_len], 
                mode='constant', 
                value=self.eos_token_id)
            for seq_len, feature in zip(seq_length, features)
        ], dim=0)
        labels = torch.cat([
            torch.nn.functional.pad(torch.cat(
                [
                    torch.full_like(feature['input_ids'], fill_value=-100), 
                    feature['labels'],
                    torch.zeros([feature['input_ids'].shape[0], 1], dtype=feature['input_ids'].dtype)
                ], dim=-1), 
                pad=[0, max_length - seq_len - 1], 
                mode='constant', 
                value=-100)
            for seq_len, feature in zip(seq_length, features)
        ], dim=0)

        return { 
            'input_ids':input_ids, 
            'labels':labels,
        }

def preprocess(example:Dict[str, str], tokenizer: GPTTokenizer) -> Dict[str, torch.Tensor]:
    return {
        "input_ids": tokenizer(example['i'], add_special_tokens=False), 
        "labels": tokenizer(example['o'], add_special_tokens=False)
        }

def get_dataset_fn(
    train_args:TrainArguments, 
    data_args:DataArguments, 
    eval_args:EvalArguments,
    tokenizer: GPTTokenizer,
    **kwargs) -> Dict[str, Dataset]:

    with train_args.main_process_first():
        train_datasets = load_dataset(
            path="json",
            data_files=data_args.dataset,
        )['train']
        eval_datasets = load_dataset(
            path="json",
            data_files=eval_args.eval_dataset,
        )['train']
        train_datasets = train_datasets.map(
            partial(preprocess, tokenizer=tokenizer),
            num_proc=16,
            remove_columns=train_datasets.column_names,
        )
        eval_datasets = eval_datasets.map(
            partial(preprocess, tokenizer=tokenizer),
            num_proc=16,
            remove_columns=eval_datasets.column_names,
        )
        train_datasets.set_format("torch")
        eval_datasets.set_format("torch")

    return {
        'train_dataset': train_datasets,
        'eval_dataset': eval_datasets
    }

def prepare_optimizer_fn(optim_cls:Optimizer, model:GPTModel, **kwargs) -> Optimizer:
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
        "get_model_fn": get_model_fn,
        "get_dataset_fn": get_dataset_fn,
        "dataset_num_proc": 16,
        "batched": False,
        "train_collate_fn": DataCollatorwithGPT(),
        "eval_collate_fn": DataCollatorwithGPT(),
        "prepare_optimizer_fn": prepare_optimizer_fn,
        "prepare_lr_scheduler_fn": prepare_lr_scheduler_fn,
        "lr_scheduler_kwargs": {
            "min_lr_rate": 0.1,
        },     
    }
    run_train(kwargs)

if __name__ == "__main__":
    main()