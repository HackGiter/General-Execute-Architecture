import os
import json
from functools import partial
from typing import Callable, Literal, Union, Dict, List, Any

import torch
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset, concatenate_datasets

from ..args import TrainArguments, EvalArguments, DataArguments, ModelArguments
from ..utils.logging import get_logger
from ..model.template import Template, MODEL_TEMPLATES

from .dataclass import Profile, Sequences, PROFILE_CLASSES
from .align import ALIGN_FUNCTIONS
from .sequence import SEQUENCE_PROFILES, get_sequences_from_config

logger = get_logger(__name__)

PROFILES4DATASET: Dict[str, Dict[str, Profile]] = {
    "sequence": SEQUENCE_PROFILES
}

CONFIG4PROFILES: Dict[str, Callable] = {
    "sequence": get_sequences_from_config
}

def load_datasets(profile:Union[List[Profile], Profile], **kwargs) -> Dataset:
    if profile.load_from == "hf":
        dataset = load_dataset(
            profile.path,
            num_proc=kwargs.get("num_proc", None)
            )
    elif profile.load_from == "file":
        _, ext = os.path.splitext(profile.path)
        dataset = load_dataset(
            path=f"{ext.replace(".", "")}" if ext != ".jsonl" else "json",
            data_dir=os.path.dirname(profile.path),
            data_files=profile.path,
            num_proc=kwargs.get("num_proc", None)
        )
    else:
        raise NotImplementedError
    if profile.split is not None:
        dataset = dataset[profile.split] if not isinstance(profile.split, List) else concatenate_datasets([dataset[item] for item in profile.split])
    profile.column_names = dataset.column_names
    return dataset

def align_dataset(dataset:Dataset, profile:Profile, desc="Dataset Aligning", **kwargs) -> Dataset:
    if isinstance(profile, Sequences):
        return dataset.map(
            partial(
                ALIGN_FUNCTIONS[profile.dtype], 
                contexts=profile.contexts, 
                instructions=profile.instructions, 
                responses=profile.responses, 
                conversations=profile.conversations,
                roles=profile.roles,
                ),
            remove_columns=profile.column_names,
            desc=desc,
            **kwargs,
        )
    else:
        NotImplemented

def preprocess_dataset( 
        dataset: Dataset,
        tokenizer:AutoTokenizer = None, 
        sys_prompt:str = None,
        max_length:int = None,
        eos_last:bool = True,
        template:Template = None,
        desc:str = "Dataset Preprocessing",
        **kwargs) -> Dataset:
    return dataset.map(
        partial(
            template.get_prompts,
            sys_prompt=sys_prompt,
            tokenizer=tokenizer,
            max_length=max_length,
            eos_last=eos_last,
        ),
        remove_columns=dataset.column_names,
        desc=desc,
        **kwargs,
    )

def log_print_example(
        dataset:Dataset, 
        tokenizer: AutoTokenizer,
        name: str,
    ) -> None:
    example = tokenizer.batch_decode(dataset[0]["input_ids"],add_special_tokens=False,)[0]
    logger.info(f"\n{name} Example:\n{example}")

def get_dataset_kwargs(train_args: TrainArguments, **kwargs) -> Dict[str, Any]:
    return {
        "num_proc": train_args.get("dataset_num_proc", kwargs.get("dataset_num_proc", 4)),
        "batched": train_args.get("batched", kwargs.get("batched", False)),
    }

def get_dataset(
        train_args:TrainArguments, 
        model_args:ModelArguments, 
        data_args:DataArguments, 
        eval_args:EvalArguments,
        tokenizer:AutoTokenizer,
        max_length:int = None,
        eos_last:bool = True,
        **kwargs) -> Dict[str, Dataset]:
    
    if data_args.dataset_dir is not None:
        with open(os.path.join(data_args.dataset_dir, "dataset_info.json"), 'r') as f:
            dataset_configs = json.load(f)
    else:
        dataset_configs = None

    template = MODEL_TEMPLATES[model_args.model] if model_args.model in MODEL_TEMPLATES else None
    train_dataset_profiles = [
        PROFILES4DATASET[it1].get(
            it2, 
            CONFIG4PROFILES[it1](it2, dataset_configs[it2])
        ) for it1, it2 in zip(data_args.dataset_type, data_args.dataset)]
    eval_dataset_profiles = [
        PROFILES4DATASET[it1].get(
            it2, 
            CONFIG4PROFILES[it1](it2, dataset_configs[it2])
        ) for it1, it2 in zip(eval_args.eval_dataset_type, eval_args.eval_dataset)] if eval_args.eval_dataset is not None else None

    dataset_kwargs = get_dataset_kwargs(train_args, **kwargs)
    with train_args.main_process_first():
        train_datasets, eval_datasets = [], None
        for profile in train_dataset_profiles:

            logger.info(f"Loading {profile.name}")
            train_dataset = load_datasets(profile, **dataset_kwargs)
            
            if template is not None:
                train_dataset = align_dataset(
                    dataset=train_dataset, 
                    profile=profile,
                    desc=f"Aligning {profile.name}",
                    **dataset_kwargs)
                
                train_dataset = preprocess_dataset(
                    dataset=train_dataset, 
                    tokenizer=tokenizer, 
                    sys_prompt=template.sys_prompt if template is not None and data_args.with_sys_prompt else None, 
                    max_length=max_length,
                    eos_last=eos_last,
                    template=template,
                    desc=f"Preprocessing {profile.name}",
                    **dataset_kwargs
                    )
            else:
                train_dataset = train_dataset.map(
                    partial(tokenizer.apply_chat_template,),
                    remove_columns=train_dataset.column_names,
                    desc=f"Preprocessing {profile.name}",
                    **kwargs,
                )
            
            log_print_example(
                dataset=train_dataset,
                tokenizer=tokenizer, 
                name=profile.name,
            )
            train_datasets.append(train_dataset)
        train_datasets:Dataset = concatenate_datasets(train_datasets)

        if eval_dataset_profiles is not None:
            eval_datasets = []
            for profile in eval_dataset_profiles:
                logger.info(f"Loading {profile.name}")
                eval_dataset = load_datasets(profile, **dataset_kwargs)
                eval_dataset = align_dataset(
                    dataset=eval_dataset, 
                    profile=profile, 
                    desc=f"Aligning {profile.name}", 
                    **dataset_kwargs)
                eval_dataset = preprocess_dataset(
                    dataset=eval_dataset, 
                    tokenizer=tokenizer, 
                    sys_prompt=template.sys_prompt if data_args.with_sys_prompt else None, 
                    max_length=max_length,
                    eos_last=eos_last,
                    template=template,
                    desc=f"Preprocessing {profile.name}",
                    **dataset_kwargs,
                    )
                eval_datasets.append(eval_dataset)
            eval_datasets:Dataset = concatenate_datasets(eval_dataset)
        else:
            eval_datasets = None

        postprocess_dataset:Callable = kwargs.pop("postprocess_dataset", None)
        if postprocess_dataset is not None:
            postprocess_remove_columns = kwargs.get("postprocess_remove_columns", False)
            postprocess_kwargs = {
                "tokenizer":tokenizer,
                "max_length":max_length,
                "eos_last":eos_last,
                "template":template,
            }
            postprocess_kwargs.update(kwargs.pop("postprocess_kwargs", {}))
            dataset_kwargs.update({k:postprocess_kwargs.pop(k) for k in dataset_kwargs if k in postprocess_kwargs})
            
            train_datasets = train_datasets.map(
                partial(postprocess_dataset, **postprocess_kwargs),
                remove_columns=postprocess_remove_columns,
                desc="Postprocessing training dataset",
                **dataset_kwargs,
            )
            eval_datasets = eval_datasets.map(
                partial(postprocess_dataset, **postprocess_kwargs),
                remove_columns=postprocess_remove_columns,
                desc="Postprocessing evaluation dataset",
                **dataset_kwargs,
            ) if eval_datasets is not None else eval_datasets

        if train_args.val_ratio > 0:
            datasets = train_datasets.train_test_split(train_args.val_ratio, shuffle=train_args.shuffle)
            train_datasets = datasets["train"]
            eval_datasets = datasets["test"]

        train_datasets.set_format("torch")
        if eval_datasets is not None:
            eval_datasets.set_format("torch")

    return {
        "train_dataset": train_datasets,
        "eval_dataset": eval_datasets
    }
