from dataclasses import dataclass
from typing import Literal, Dict, List

from .dataclass import Sequences

SEQUENCE_PROFILES: Dict[str, Sequences] = {}

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
    SEQUENCE_PROFILES[name] = Sequences(
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
