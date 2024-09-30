from dataclasses import dataclass, field
from typing import Literal, Optional

@dataclass
class DataArguments:
    """
    Aguments about data behavious: Loading, Preprocessing, Postprocessing and etc.
    """
    dataset: str = field(
        default=None,
        metadata={
            "help": "name(s) of training datasets"
        }
    )
    dataset_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "directory of training datasets"
        }
    )
    max_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "max length of sequences if task is about natural language modeling"
        }
    )
    mix_strategy: Literal['mean', 'sample', 'no'] = field(
        default='no',
        metadata={
            "help": "Strategy to mix up datasets"
        }
    )
    with_sys_prompt: bool = field(
        default=True,
        metadata={
            "help": "with system prompt or not"
        }
    )

    def __post_init__(self):
        if self.dataset is not None:
            self.dataset = [item.strip() for item in self.dataset.replace(" ", "").split(',')]
    