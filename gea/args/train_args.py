import os
from datetime import timedelta
from dataclasses import dataclass, field
from typing import Literal, Optional, Union
from contextlib import contextmanager

import torch
from transformers import enable_full_determinism, set_seed
from transformers.utils import is_torch_available, is_torch_cuda_available, is_torch_tf32_available
from transformers.trainer_utils import SchedulerType
from accelerate.state import AcceleratorState, PartialState

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
    do_debug: Optional[bool] = field(
        default=False,
        metadata={
            "help": "whether to debug or not"
        }
    )
    seed: Optional[int] = field(
        default=1234,
        metadata={
            "help": "seed for initialization and reproducible experiments"
        }
    )
    full_determinism: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Whether to call enable_full_determinism instead of set_seed for reproducibility in distributed"
                " training. Important: this will negatively impact the performance, so only use it for debugging."
            )
        }
    )
    use_seedable_sampler: Optional[bool] = field(
        default=True,
        metadata={
            "help": (
                "Whether or not use a fully seedable random sampler ([`accelerate.data_loader.SeedableRandomSampler`]). Ensures "
                "training results are fully reproducable using a different sampling technique. While seed-to-seed results "
                "may differ, on average the differences are neglible when using multiple different seeds to compare. Should "
                "also be ran with [`~utils.set_seed`] for the best results."
            )
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
    dataset_num_proc: int = field(
        default=4,
        metadata={
            "help": "number of workers for loading/processing dataset batches"
        }
    )
    load_from_cache_file: bool = field(
        default=False,
        metadata={
            "help": "whether load from dataset cache file or not"
        }
    )
    dataloader_num_workers: int = field(
        default=1,
        metadata={
            "help": "number of workers for dataloader batches"
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
    even_batches: bool = field(
        default=True,
        metadata={
            "help": "If set to `True`, in cases where the total batch size across all processes does not exactly divide the"
            " dataset, samples at the start of the dataset will be duplicated so the batch can be divided equally among"
            " all workers."
        }
    )
    non_blocking: bool = field(
        default=False,
        metadata={
            "help": "Whether to use non-blocking CUDA calls to help minimize synchronization during "
            "distributed training with prepared `DataLoader` inputs being moved to device. "
            "Best if used with `pin_memory=True` in the `TrainingArguments`. Requires accelerate "
            "v0.30.0."
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
    mixed_precision: Literal['bf16', 'fp16', 'no'] = field(
        default="no",
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
    upcast_layernorm: bool = field(
        default=False,
        metadata={"help": "Whether or not to upcast the layernorm weights in fp32."},
    )
    gradient_checkpointing: bool = field(
        default=False,
        metadata={
            "help": "Acitvate gradient checkpointing if needed"
        }
    )
    use_unsloth_gc: bool = field(
        default=False,
        metadata={"help": "Whether or not to use unsloth's gradient checkpointing."},
    )
    upcast_lmhead_output: bool = field(
        default=False,
        metadata={"help": "Whether or not to upcast the output of lm_head in fp32."},
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
        enable_full_determinism(self.seed) if self.full_determinism else set_seed(self.seed)
        if is_torch_available():
            if is_torch_cuda_available():
                self.ddp_backend = "nccl"
                if is_torch_tf32_available():
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
            else:
                self.ddp_backend = "hccl"
        
        if self.mixed_precision != "no":
            os.environ["ACCELERATE_MIXED_PRECISION"] = self.mixed_precision

        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            accelerator_state_kwargs = {}
            AcceleratorState._reset_state(reset_partial_state=True)
            accelerator_state_kwargs["backend"] = self.ddp_backend
            accelerator_state_kwargs["timeout"] = timedelta(seconds=self.ddp_timeout)
            if self.deepspeed:
                os.environ["ACCELERATE_USE_DEEPSPEED"] = "true"
            self.distributed_state = PartialState(**accelerator_state_kwargs)
            # if self.deepspeed:
            #     del os.environ["ACCELERATE_USE_DEEPSPEED"]
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

