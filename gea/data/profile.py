import os
from functools import partial
from dataclasses import dataclass
from typing import Literal, Callable, Union, Dict, List, Any

import torch
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets

from ..args import TrainArguments, EvalArguments, DataArguments, ModelArguments
from ..utils.logging import get_logger
from ..model.template import Template, MODEL_TEMPLATES

from .align import align_sequence, align_dialogue, align_multi_turn

logger = get_logger(__name__)

def get_prompts(
        example: Dict[str, Any],
        sys_prompt:str, 
        tokenizer: AutoTokenizer, 
        max_length:int = None, 
        template:Template = None,
        eos_last:bool = True,
        concatenated:bool = True,
        **kwargs) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
    instructions: List[str] = example["instructions"]
    responses: List[str] = example["responses"]
    sys_prompt = "" if sys_prompt is None else template.get_sys_prompt(sys_prompt)
    instructions = [
        template.get_inst_prompt(item.strip(), sys_prompt) if i == 0 \
            else template.get_inst_prompt(item.strip(), "")
            for i, item in enumerate(instructions)
    ]
    responses = [
        item.lstrip() if not eos_last and i == len(responses) - 1 else template.get_response(item.strip())
        for i, item in enumerate(responses)
    ]
    instructions = [
        tokenizer(
            item, 
            return_tensors="pt",
            add_special_tokens=False
        ) for item in instructions
    ]
    responses = [
        tokenizer(
            item,
            return_tensors="pt",
            add_special_tokens=False
        ) for item in responses
    ]
    if len(instructions) != 0 and len(responses) != 0:
        input_ids = torch.cat(
            [torch.cat((instr["input_ids"], label["input_ids"]), dim=-1) for instr, label in zip(instructions, responses)] + 
            ([] if len(instructions) == len(responses) else 
             ([item["input_ids"] for item in responses[len(instructions):]] if len(instructions) < len(responses) else 
              [item["input_ids"] for item in instructions[len(responses):]]))
        , dim=-1)
        labels = torch.cat(
            [torch.cat((torch.zeros_like(instr["input_ids"]), label["input_ids"]), dim=-1) for instr, label in zip(instructions, responses)] + 
            ([] if len(instructions) == len(responses) else 
             ([item["input_ids"] for item in responses[len(instructions):]] if len(instructions) < len(responses) else 
              [item["input_ids"] for item in instructions[len(responses):]]))
        , dim=-1)
        attention_mask = torch.cat(
            [torch.cat((instr["attention_mask"], label["attention_mask"]), dim=-1) for instr, label in zip(instructions, responses)] + 
            ([] if len(instructions) == len(responses) else 
             ([item["attention_mask"] for item in responses[len(instructions):]] if len(instructions) < len(responses) else 
              [item["attention_mask"] for item in instructions[len(responses):]]))
        , dim=-1)
        if max_length is not None:
            input_ids = input_ids[:, :max_length]
            labels = labels[:, :max_length]
            attention_mask = attention_mask[:, :max_length]
    else:
        instructions = instructions if len(instructions) != 0 else responses
        if concatenated:
            input_ids = torch.cat([item["input_ids"] for item in instructions], dim=-1)
            labels = input_ids
            attention_mask = torch.cat([item["attention_mask"] for item in instructions], dim=-1)
        else:
            input_ids = [item["input_ids"] for item in instructions]
            labels = None
            attention_mask = [item["attention_mask"] for item in instructions]

    return { "input_ids":input_ids, "labels":labels, "attention_mask":attention_mask }

@dataclass
class Profile:
    name: str = None
    load_from: Literal["hf", "file"] = "hf"
    formatting: List[str] = None
    path: str = None
    split: Union[str, List[str]] = None
    dataset: Union[DatasetDict, Dataset] = None
    column_names: List[str] = None

@dataclass
class Sequences(Profile):
    formatting: Literal["sequence", "dialogue", "multi-turn", "custom"] = "sequence"
    contexts: str = None
    instructions: str = None
    responses: str = None
    conversations: List[str] = None,
    roles: List[str] = None

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
            get_prompts,
            sys_prompt=sys_prompt,
            tokenizer=tokenizer,
            max_length=max_length,
            template=template,
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

ALIGN_FUNCTIONS: Dict[str, Callable] = {
    "sequence": align_sequence,
    "dialogue": align_dialogue,
    "multi-turn": align_multi_turn,
}
DATASET_PROFILES: Dict[str, Profile] = {}

def register_sequences(
    name: str,
    load_from: Literal["hf", "file"] = None,
    path: str = None,
    split: str = "train",
    formatting: Literal["sequence", "dialogue", "multi-turn"] = "sequence",
    contexts: List[str] = None,
    instructions: List[str] = None,
    responses: List[str] = None,
    conversations: List[str] = None,
    roles: List[str] = None,
) -> None:
    DATASET_PROFILES[name] = Sequences(
        name=name, 
        load_from=load_from, 
        path=path, 
        split=split, 
        formatting=formatting,
        contexts=contexts, 
        instructions=instructions, 
        responses=responses, 
        conversations=conversations, 
        roles=roles, 
    )

register_sequences(
    name="Magicoder-Evol-Instruct-110K",
    load_from="file",
    path="/data/datasets/Magicoder-Evol-Instruct-110K/data-evol_instruct-decontaminated.jsonl",
    split="train",
    formatting="dialogue",
    contexts=[],
    instructions=["instruction"],
    responses=["response"],
    conversations=None,
    roles=None,
)

register_sequences(
    name="ultrachat_200k",
    load_from="hf",
    path="/data/datasets/ultrachat_200k",
    split=["train_sft", "test_sft", "train_gen", "test_gen"],
    formatting="multi-turn",
    contexts=[],
    instructions=["messages"],
    responses=["messages"],
    conversations=["content", "role"],
    roles=["user", "assistant"],
)

register_sequences(
    name="ShareGPT_Vicuna_unfiltered",
    load_from="file",
    path="/data/lhz/datasets/ShareGPT_Vicuna_unfiltered/ShareGPT_V4.3_unfiltered_cleaned_split.json",
    split="train",
    formatting="multi-turn",
    contexts=[],
    instructions=["conversations"],
    responses=["conversations"],
    conversations=["value", "from"],
    roles=["human", "gpt"]
)

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
    
    template = MODEL_TEMPLATES[model_args.model] if model_args.model in MODEL_TEMPLATES else None
    train_dataset_profiles = [DATASET_PROFILES[item] for item in data_args.dataset]
    if eval_args.eval_dataset is not None:
        eval_dataset_profiles = [DATASET_PROFILES[item] for item in eval_args.eval_dataset]

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

        if eval_args.eval_dataset is not None:
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
            if eval_args.eval_dataset is not None:
                eval_datasets = eval_datasets.map(
                    partial(postprocess_dataset, **postprocess_kwargs),
                    remove_columns=postprocess_remove_columns,
                    desc="Postprocessing evaluation dataset",
                    **dataset_kwargs,
                )

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
