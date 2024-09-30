from dataclasses import dataclass
from typing import Literal, Union, List

from datasets import Dataset, DatasetDict

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