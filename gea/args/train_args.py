import os
from datetime import timedelta
from dataclasses import dataclass, field
from typing import Literal, Optional, Union
from contextlib import contextmanager

from transformers.trainer_utils import SchedulerType
from accelerate import PartialState

from ..utils.callback import StateStrategy, Optim

@dataclass
class TrainArguments:
    """
    Aguments about training: Optimizer, Learning rate scheduler, weight decay and etc.
    """
    do_train: Optional[bool] = field(
        default=True,
        metadata={
            "help": "whether to trai or not"
        }
    )
    seed: Optional[int] = field(
        default=1234,
        metadata={
            "help": "seed for initialization and reproducible experiments"
        }
    )
    epochs: Optional[float] = field(
        default=1.0,
        metadata={
            "help": "maximum epoches of model training"
        }
    )
    max_steps: Optional[int] = field(
        default=0,
        metadata={
            "help": "max steps of training procedure"
        }
    )
    logging_steps: Optional[int] = field(
        default=10,
        metadata={
            "help": "logging steps of training procedure"
        }
    )
    save_steps: Optional[int] = field(
        default=10,
        metadata={
            "help": "save steps of training procedure"
        }
    )
    save_strategy: Union[StateStrategy, str] = field(
        default='epoch',
        metadata={
            "help": "save strategies: no, steps, epoch"
        }
    )
    eval_steps: Optional[int] = field(
        default=0,
        metadata={
            "help": "evaluation steps of training procedure"
        }
    )
    eval_strategy: Union[StateStrategy, str] = field(
        default='no',
        metadata={
            "help": "evaluation strategies: no, steps, epoch"
        }
    )
    val_ratio: Optional[float] = field(
        default=-1,
        metadata={
            "help": "the ratio of evaluation data in the training dataset ranging from 0.0 to 1.0"
        }
    )
    per_device_train_batch_size: Optional[int] = field(
        default=1,
        metadata={
            "help": "training batch size per device"
        }
    )
    per_device_eval_batch_size: Optional[int] = field(
        default=1,
        metadata={
            "help": "evaluation batch size per device"
        }
    )
    shuffle: bool = field(
        default=True,
        metadata={
            "help": "whether to shuffle the dataloader"
        }
    )
    dataloader_num_workers: int = field(
        default=1,
        metadata={
            "help": "number of workers for loading/processing dataset batches"
        }
    )
    dataloader_prefetch_factor: int = field(
        default=1,
        metadata={
            "help": "prefetch numbers of data batches: num_workers * prefetch_factor"
        }
    )
    dataloader_pin_memory: bool = field(
        default=True,
        metadata={
            "help": "whether pin memory of data batches"
        }
    )
    dataloader_drop_last: bool = field(
        default=False,
        metadata={
            "help": "drop last part of dataloader"
        }
    )
    project: Optional[str] = field(
        default=None,
        metadata={
            "help": "directory for training project"
        }
    )
    tensorboard_project: Optional[str] = field(
        default=None,
        metadata={
            "help": "directory for tensorboard project"
        }
    )
    optim: Union[Optim, str] = field(
        default='adamw',
        metadata={
            "help": "name of optimizer"
        }
    )
    opt_kwargs: Optional[str] = field(
        default=None,
        metadata={
            "help": "other arguments of optimizer constructed as dict obj"
        }
    )
    lr_scheduler: Union[SchedulerType, str] = field(
        default="cosine",
        metadata={
            "help": "name of learning rate scheduler"
        }
    )
    lr: float = field(
        default=1e-4,
        metadata={
            "help": "learning rate"
        }
    )
    warmup_steps: Optional[int] = field(
        default=0,
        metadata={
            "help": "warm-up steps of learning rate"
        }
    )
    warmup_ratio: float = field(
        default=0.0,
        metadata={
            "help": "warm-up ratio refers to the proportion of warming-up steps to the total number of steps"
        }
    )
    lr_scheduler_kwargs: Optional[str] = field(
        default=None,
        metadata={
            "help": "other keyword arguments of learning rate scheduler"
        }
    )
    weight_decay: float = field(
        default=0.0,
        metadata={
            "help": "weight decay rate ranging from 0 to 1.0"
        }
    )
    max_grad_norm: float = field(
        default=-1.0,
        metadata={
            "help": "maximum gradient norm clip"
        }
    )
    mixed_precision: Literal['bf16', 'fp16'] = field(
        default=None,
        metadata={
            "help": "training with bfloat16/float16 precision"
        }
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={
            "help": "gradient accumulation steps of training"
        }
    )
    gradient_checkpointing: bool = field(
        default=False,
        metadata={
            "help": "Acitvate gradient checkpointing if needed"
        }
    )
    resume_from_checkpoint: str = field(
        default=None,
        metadata={
            "help": "directory with checkpoint states necessary for resuming training"
        }
    )
    save_total_limit: int = field(
        default=None,
        metadata={
            "help": "the total limit of checkpoints saved"
        }
    )
    ddp_backend: str = field(
        default="nccl",
        metadata={
            "help": "The backend to use for distributed training. Must be one of `'nccl'`, `'mpi'`, `'ccl'`, `'gloo'`, `'hccl'`."
        }
    )
    ddp_timeout: int = field(
        default=1800,
        metadata={
            "help": "distributed data parallel timeout"
        }
    )
    deepspeed: bool = field(
        default=False,
        metadata={
            "help": "whether deepspped on or not"
        }
    )

    def __post_init__(self):
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            accelerator_state_kwargs = {}
            accelerator_state_kwargs["backend"] = self.ddp_backend
            accelerator_state_kwargs["timeout"] = timedelta(seconds=self.ddp_timeout)
            if self.deepspeed:
                os.environ["ACCELERATE_USE_DEEPSPEED"] = "true"
            self.distributed_state = PartialState(**accelerator_state_kwargs)
            if self.deepspeed:
                del os.environ["ACCELERATE_USE_DEEPSPEED"]
        else:
            self.distributed_state = None

    @contextmanager
    def main_process_first(self):
        if self.distributed_state is None:
            yield
        else:
            with self.distributed_state.main_process_first():
                yield

    def __getitem__(self, key:str):
        return getattr(self, key)        
    
    def get(self, key:str, default=None):
        return getattr(self, key, default)

