import json
import copy
import time
import dataclasses
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Literal, Union, Dict, List, Any
from tqdm.auto import tqdm

import torch
from transformers import (
    AutoModel, 
    AutoTokenizer,
)
from transformers.trainer_utils import SchedulerType

from .logging import get_logger

logger = get_logger(__name__)


class StateStrategy(str, Enum):
    NO = "no"
    STEPS = "steps"
    EPOCH = "epoch"

class Optim(str, Enum):
    """
    Stores the acceptable string identifiers for optimizers.
    """
    ADAMW = "adamw"
    SGD = "sgd"
    ADAGRAD = "adagrad"

@dataclass
class GeneralState:
    global_step: int = 0
    logging_steps: int = 0
    log_history: List[Dict[str, float]] = None
    is_local_process_zero: bool = True
    is_world_process_zero: bool = True
    state_callbacks: List["StateCallback"] = None

    def __post_init__(self):
        if self.log_history is None:
            self.log_history = []
        if self.state_callbacks is None:
            self.state_callbacks = []
        else:
            state_callbacks = []
            for callback in self.state_callbacks:
                name = callback.__class__.__name__
                if name not in state_callbacks:
                    state_callbacks.append(name)
            self.state_callbacks = state_callbacks

    def save_to_json(self, json_path: str):
        """Save the content of this instance in JSON format inside `json_path`."""
        json_string = json.dumps(dataclasses.asdict(self), indent=2, sort_keys=True) + "\n"
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(json_string)

    @classmethod
    def load_from_json(cls, json_path: str):
        """Create an instance from the content of `json_path`."""
        with open(json_path, "r", encoding="utf-8") as f:
            text = f.read()
        return cls(**json.loads(text))

@dataclass
class TrainState(GeneralState):
    epochs: Union[float, int] = None
    global_epoch: int = 0
    num_samples: int = 0
    max_steps: int = 0
    eval_steps: int = 0
    save_steps: int = 0
    train_batch_size: int = 0
    gradient_accumulation_steps: int = 0
    cur_loss: Union[torch.Tensor, float] = None
    train_loss: float = 0.0
    eval_loss: float = 0.0
    cur_flops: float = 0
    total_flops: float = 0
    best_metric: Optional[float] = None
    best_model_checkpoint: Optional[str] = None
    is_hyper_param_search: bool = False
    trial_name: str = None
    trail_params: Dict[str, Union[str, float, int, bool]] = None
    optim: Union[Optim, str] = None
    lr: float = None
    lr_scheduler: Union[SchedulerType, str] = None
    warmup_steps: int = None
    weight_decay: float = 0.0
    max_grad_norm: float = None

    eval_strategy: Union[StateStrategy, Literal['no', 'step', 'epoch']] = StateStrategy.NO
    save_strategy: Union[StateStrategy, Literal['no', 'step', 'epoch']] = StateStrategy.STEPS
    # logging_strategy: Union[StateStrategy, Literal['no', 'step', 'epoch']] = StateStrategy.STEPS

    wrapped: bool = False
    should_log: bool = False
    should_eval: bool = False
    should_save: bool = False
    should_stop: bool = False

    start_time: float = 0
    end_time: float = 0

    @property
    def hparams(self) -> Dict[str, Any]:
        return {
            "epochs": self.epochs,
            "num_samples": self.num_samples,
            "max_steps": self.max_steps,
            "batch_size": self.train_batch_size,
            "optim": self.optim,
            "lr_scheduler": self.lr_scheduler,
            "lr": self.lr,
            "warmup_steps": self.warmup_steps,
            "weight_decay": self.weight_decay,
            "max_grad_norm": self.max_grad_norm,
        }
    
    def get_train_metric(self, prefix:str = None):
        prefix = "" if prefix is None else prefix + "_"
        runtime = self.end_time - self.start_time
        return {
            f"{prefix}runtime": round(runtime, 4),
            f"{prefix}samples_per_second": round(self.num_samples / runtime, 3),
            f"{prefix}steps_per_second": round(self.max_steps / runtime, 3),
            f"{prefix}total_flos": self.total_flops
        }

class StateCallback:
    """
    Callback object for recording the state during training, evaluation, inference and etc
    """
    def on_init(self, **kwargs):
        pass

    def on_train_begin(self, **kwargs):
        pass

    def on_train_end(self, **kwargs):
        pass

    def on_eval_begin(self, **kwargs):
        pass

    def on_eval_end(self, **kwargs):
        pass

    def on_infer_begin(self, **kwargs):
        pass

    def on_infer_end(self, **kwargs):
        pass

    def on_epoch_begin(self, **kwagrs):
        pass

    def on_epoch_end(self, **kwargs):
        pass

    def on_step_begin(self, **kwargs):
        pass

    def on_step_end(self, **kwargs):
        pass

    def on_eval_step(self, **kwargs):
        pass

    def on_log(self, **kwargs):
        pass

    def on_save(self, **kwargs):
        pass

    def on_finish(self, **kwargs):
        pass

class CallbackHandler(StateCallback):

    def __init__(
            self, 
            callbacks: List[StateCallback], 
            model: AutoModel, 
            tokenizer: AutoTokenizer, 
            **kwargs) -> None:
        self.callbacks = []
        for cb in callbacks:
            self.add_callback(cb)
        self.model = model
        self.tokenizer = tokenizer
        for key, value in kwargs.items():
            setattr(self, key, value)

    def add_callback(self, callback: StateCallback):
        cb = callback() if isinstance(callback, type) else callback
        cb_class = callback if isinstance(callback, type) else callback.__class__
        if cb_class in [c.__class__ for c in self.callbacks]:
            logger.warning(
                f"You are adding a {cb_class} to the callbacks of this Trainer, but there is already one. The current"
                + "list of callbacks is\n:"
                + self.callback_list
            )
        self.callbacks.append(cb)

    def pop_callback(self, callback: StateCallback):
        if isinstance(callback, type):
            for cb in self.callbacks:
                if isinstance(cb, callback):
                    self.callbacks.remove(cb)
                    return cb
        else:
            for cb in self.callbacks:
                if cb == callback:
                    self.callbacks.remove(cb)
                    return cb

    def remove_callback(self, callback: StateCallback):
        if isinstance(callback, type):
            for cb in self.callbacks:
                if isinstance(cb, callback):
                    self.callbacks.remove(cb)
                    return
        else:
            self.callbacks.remove(callback)

    @property
    def callback_list(self):
        return "\n".join(cb.__class__.__name__ for cb in self.callbacks)

    def on_init(self, **kwargs):
        return self.call_event("on_init", **kwargs)

    def on_train_begin(self, **kwargs):
        return self.call_event("on_train_begin", **kwargs)

    def on_train_end(self, **kwargs):
        return self.call_event("on_train_end", **kwargs)

    def on_eval_begin(self, **kwargs):
        return self.call_event("on_eval_begin", **kwargs)

    def on_eval_end(self, **kwargs):
        return self.call_event("on_eval_end", **kwargs)

    def on_infer_begin(self, **kwargs):
        return self.call_event("on_infer_begin", **kwargs)

    def on_infer_end(self, **kwargs):
        return self.call_event("on_infer_end", **kwargs)

    def on_epoch_begin(self, **kwargs):
        return self.call_event("on_epoch_begin", **kwargs)

    def on_epoch_end(self, **kwargs):
        return self.call_event("on_epoch_end", **kwargs)

    def on_step_begin(self, **kwargs):
        return self.call_event("on_step_begin", **kwargs)

    def on_step_end(self, **kwargs):
        return self.call_event("on_step_end", **kwargs)
    
    def on_eval_step(self, **kwargs):
        return self.call_event("on_eval_step", **kwargs)

    def on_log(self, **kwargs):
        return self.call_event("on_log", **kwargs)

    def on_save(self, **kwargs):
        return self.call_event("on_save", **kwargs)

    def on_finish(self, **kwargs):
        return self.call_event("on_finish", **kwargs)

    def call_event(self, event, **kwargs):
        control = None
        for callback in self.callbacks:
            result = getattr(callback, event)(
                model=self.model,
                tokenizer=self.tokenizer,
                **kwargs,
            )
            # A Callback can skip the return of `control` if it doesn't change it.
            if result is not None:
                control = result
        return control

class TrainStateCallback(StateCallback):

    def __init__(self) -> None:
        self.training_bar = None
        self.evaluation_bar = None
        super().__init__()

    def on_train_begin(self, state:TrainState, **kwargs):
        if state.is_world_process_zero:
            state.start_time = time.time()
            self.training_bar = tqdm(total=state.max_steps, dynamic_ncols=True)
        state.wrapped = True
        self.current_step = 0

    def on_eval_begin(self, state:TrainState, eval_dataloader=None, **kwargs):
        state.should_log = True
        if state.is_world_process_zero and eval_dataloader is not None:
            self.evaluation_bar = tqdm(total=len(eval_dataloader), leave=self.training_bar is None, dynamic_ncols=True)

    def on_init(self, state:TrainState, **kwargs):
        if state.is_world_process_zero:
            logger.info("Trainer initialize")

    def on_epoch_end(self, state:TrainState, **kwagrs):
        state.global_epoch += 1

    def on_step_begin(self, state:TrainState, sync_on:bool, **kwargs):
        state.should_log = False
        state.should_eval = False
        state.should_save = False
    
    def on_step_end(self, state:TrainState, loss:Union[torch.Tensor, float], flops:float, sync_on:bool, **kwargs):
        state.global_step += sync_on
        state.cur_flops += flops
        if sync_on:
            if state.is_world_process_zero:
                self.training_bar.update(state.global_step - self.current_step)
            state.loss = (state.loss if isinstance(state.loss, loss) else state.loss.item()) + loss
            self.current_step = state.global_step
        if (
            self.current_step == 1
            or self.current_step % state.logging_steps == 0
        ):
            state.should_log = True
        if (
            state.eval_steps > 0
            and self.current_step % state.eval_steps == 0
        ):
            state.should_eval = True
        if (
            state.save_steps > 0
            and self.current_step % state.save_steps == 0
        ):
            state.should_save = True  
        state.should_stop = (state.global_step == state.max_steps)
        state.should_save = (state.should_save or state.should_stop)

    def on_eval_step(self, state:TrainState, **kwargs):
        if state.is_world_process_zero:
            if self.evaluation_bar is not None:
                self.evaluation_bar.update(1)

    def on_log(self, state:TrainState, logs:Dict[str, Any], **kwargs):
        state.should_log = False
        if "train_loss" in logs:
            state.cur_loss -= state.cur_loss
            state.loss = logs["train_loss"]
        if "train_flops" in logs:
            state.cur_flops = 0
            state.total_flops += logs["train_flops"]
        if "eval_loss" in logs:
            state.eval_loss = logs["eval_loss"]
        state.log_history.append({**logs, **{"step": state.global_step}})
        if state.is_world_process_zero and self.training_bar is not None:
            logs = copy.deepcopy(logs)
            if "epoch" in logs:
                logs["epoch"] = round(logs["epoch"], 2)
            self.training_bar.write(str(logs))
            
    def on_save(self, state:TrainState, **kwagrs):
        state.should_save = False

    def on_eval_end(self, state:TrainState, **kwargs):
        state.should_log = True
        state.should_eval = False
        if state.is_world_process_zero:
            if self.evaluation_bar is not None:
                self.evaluation_bar.close()
            self.evaluation_bar = None

    def on_train_end(self, state:TrainState, **kwargs):
        if state.is_world_process_zero:
            self.training_bar.close()
            self.training_bar = None
            state.end_time = time.time()
