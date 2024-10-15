from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ModelArguments:
    """
    Aguments about model: Loading, Construction, Hyper parameters and etc.
    """
    model: Optional[str] = field(
        default=None,
        metadata={
            "help": "name or path of model in huggingface or self-defined"
        }
    )
    pretrained: Optional[bool] = field(
        default=False,
        metadata={
            "help": "initialized from pretrained model or not"
        }
    )
    path: str = field(
        default=None,
        metadata={
            "help": "name or path of model in huggingface or self-defined",
        }
    )
    low_cpu_mem_usage: bool = field(
        default=True,
        metadata={"help": "Whether or not to use memory-efficient model loading."},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether or not to use one of the fast tokenizer (backed by the tokenizers library)."},
    )