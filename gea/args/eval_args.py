from dataclasses import dataclass, field
from typing import Literal, Optional

@dataclass
class EvalArguments:
    """
    Aguments about evaluation: dataset, metrics, inference and etc
    """
    do_eval: Optional[bool] = field(
        default=False,
        metadata={
            "help": "do evaluation during training or not"
        }
    )

    eval_dataset: Optional[str] = field(
        default=None,
        metadata={
            "help": "name or path of evaluation dataset"
        }
    )
    eval_dataset_type: str = field(
        default="sequence",
        metadata={
            "help": "type of datasets: sequence, image, etc"
        }
    )

    def __post_init__(self):
        if self.eval_dataset is not None:
            self.eval_dataset = [item.strip() for item in self.eval_dataset.split(",")]

        if self.eval_dataset_type is not None:
            self.eval_dataset_type = [item.strip() for item in self.eval_dataset_type.split(",")]
    