from dataclasses import dataclass, field
from typing import Literal, Optional

@dataclass
class EvalArguments:
    """
    Aguments about evaluation: dataset, metrics, inference and etc
    """
    eval_dataset: Optional[str] = field(
        default=None,
        metadata={
            "help": "name or path of evaluation dataset"
        }
    )

    def __post_init__(self):
        if self.eval_dataset is not None:
            self.eval_dataset = [item.strip() for item in self.eval_dataset.replace(" ", "").split(',')]
    