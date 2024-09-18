import os
import math
from contextlib import contextmanager
from typing import Callable, Mapping, Union, Tuple, Dict, List, Any, get_origin, get_args

import numpy as np

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.allow_tf32 = True
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from datasets import Dataset, IterableDataset, DatasetDict
from transformers import (
    AutoModel,
    AutoTokenizer,
    get_scheduler,
)
from accelerate import Accelerator, skip_first_batches
from accelerate.utils import LoggerType, DistributedType

from  ..args import TrainArguments
from ..utils.callback import CallbackHandler, TrainStateCallback, TrainState, Optim, StateStrategy
from ..utils.integration import TensorBoardCallback
from ..utils.tools import rotate_checkpoints
from ..utils.logging import get_logger

from transformers.trainer_utils import enable_full_determinism
from transformers.trainer_pt_utils import get_model_param_count

logger = get_logger(__name__)

OPTIMIZERS:Dict[str, Optimizer] = {
    Optim.ADAGRAD: torch.optim.Adagrad,
    Optim.ADAMW: torch.optim.AdamW,
    Optim.SGD: torch.optim.SGD,
}

class Trainer:
    """
    Scheduler for training including optimizer, lr scheduler, loss calculation and etc
    """
    def __init__(
            self,
            model: AutoModel,
            tokenizer: AutoTokenizer,
            train_args: TrainArguments,
            train_dataset: Dataset = None,
            eval_dataset: Union[Dataset, DatasetDict] = None,
            **kwargs,
            ) -> None:
        enable_full_determinism(train_args.seed)
        self.train_args = train_args

        self.accelerator = Accelerator(
            mixed_precision=self.train_args.mixed_precision,
            gradient_accumulation_steps=self.train_args.gradient_accumulation_steps,
            log_with=LoggerType.TENSORBOARD if self.train_args.tensorboard_project is not None else LoggerType.WANDB,
            project_dir=self.train_args.project,
        )
        if self.train_args.tensorboard_project is not None:
            if self.accelerator.is_main_process:
                logger.info("Tensoboard tracker initialize")
            self.accelerator.init_trackers(project_name=self.train_args.tensorboard_project)
        self.accelerator.free_memory()

        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_datasets = eval_dataset

        callbacks = ([TensorBoardCallback] if train_args.tensorboard_project is not None else []) + [TrainStateCallback]
        callbacks += kwargs.pop("callbacks", [])
        self.callback_handler = CallbackHandler(callbacks, model, tokenizer)
        self.state = TrainState(
            epochs=self.train_args.epochs,
            max_steps=self.train_args.max_steps,
            logging_steps=self.train_args.logging_steps,
            save_steps=self.train_args.save_steps,
            eval_steps=self.train_args.eval_steps,
            train_batch_size=self.train_args.per_device_train_batch_size * self.accelerator.num_processes,
            total_train_batch_size=self.train_args.per_device_train_batch_size * self.accelerator.num_processes * self.train_args.gradient_accumulation_steps,
            is_local_process_zero=self.accelerator.is_local_main_process,
            is_world_process_zero=self.accelerator.is_main_process,
            optim=self.train_args.optim,
            lr=self.train_args.lr,
            lr_scheduler=self.train_args.lr_scheduler,
            warmup_steps=self.train_args.warmup_steps,
            weight_decay=self.train_args.weight_decay,
            max_grad_norm=self.train_args.max_grad_norm,
            state_callbacks=self.callback_handler.callbacks,
            save_strategy=self.train_args.save_strategy,
            eval_strategy=self.train_args.eval_strategy,
        )
        self.callback_handler.on_init(state=self.state)

        self.prepare_train_kwargs(kwargs)

    def prepare_train_kwargs(self, kwargs:Dict[str, Any]) -> None:
        self.train_data_collator = kwargs.pop("train_collate_fn", None)
        self.eval_data_collator = kwargs.pop("eval_collate_fn", self.train_data_collator)

        self.train_dataloader = None
        self.eval_dataloaders = None
        self.train_sampler = kwargs.pop("train_sampler", None)
        self.eval_sampler = kwargs.pop("eval_sampler", None)
        self.optimizer = kwargs.pop("optimizer", None)
        self.lr_scheduler = kwargs.pop("lr_scheduler", None)

        self.kwargs = kwargs

    def calibrate_train_state(self) -> None:
        if self.train_dataloader is None:
            self.get_train_dataloader()
        steps_per_epoch = (len(self.train_dataloader) // self.train_args.gradient_accumulation_steps) // self.accelerator.num_processes
        if self.state.max_steps <= 0:
            self.state.max_steps = math.ceil(steps_per_epoch * self.train_args.epochs)
        if self.train_args.warmup_ratio > 0 and self.state.warmup_steps == 0:
            self.state.warmup_steps = math.ceil(self.state.max_steps * self.train_args.warmup_ratio)
        if isinstance(self.state.epochs, float):
            self.state.epochs = math.ceil(self.state.max_steps / steps_per_epoch)
        if self.train_args.eval_strategy == StateStrategy.EPOCH:
            self.state.eval_steps = steps_per_epoch
        elif self.train_args.eval_strategy == StateStrategy.NO:
            self.state.eval_steps = -1
        if self.train_args.save_strategy == StateStrategy.EPOCH:
            self.state.save_steps = steps_per_epoch
        elif self.train_args.save_strategy == StateStrategy.NO:
            self.state.save_steps = -1
        self.state.num_examples = self.state.max_steps * self.state.train_batch_size * self.train_args.gradient_accumulation_steps

    def get_train_dataloader(self) -> None:
        if self.train_dataloader is None:
            dataloader_params = {
                "batch_size": self.train_args.per_device_train_batch_size,
                "collate_fn": self.train_data_collator,
                "num_workers": self.train_args.dataloader_num_workers,
                "prefetch_factor": self.train_args.dataloader_prefetch_factor,
                "pin_memory": self.train_args.dataloader_pin_memory,
                "sampler": self.train_sampler,
                "shuffle": self.train_args.shuffle,
                "drop_last": self.train_args.dataloader_drop_last,
            }
            self.train_dataloader = DataLoader(self.train_dataset, **dataloader_params)
            
    def get_eval_dataloaders(self) -> None:
        if self.eval_dataloaders is None and self.eval_datasets is not None:
            dataloader_params = {
                "batch_size": self.train_args.per_device_eval_batch_size,
                "collate_fn": self.eval_data_collator,
                "num_workers": self.train_args.dataloader_num_workers,
                "prefetch_factor": self.train_args.dataloader_prefetch_factor,
                "pin_memory": self.train_args.dataloader_pin_memory,
                "sampler": self.eval_sampler,
                "shuffle": False,
                "drop_last": self.train_args.dataloader_drop_last,
            }
            if isinstance(self.eval_datasets, DatasetDict):
                self.eval_dataloaders = {}
                for key, value in self.eval_datasets.items():
                    self.eval_dataloaders[key] = DataLoader(value, **dataloader_params)
            else:
                self.eval_dataloaders = DataLoader(self.eval_datasets, **dataloader_params)

    def prepare_model(self, **kwargs) -> None:
        if not self.state.wrapped:
            self.model.train()
            prepare_model_fn = self.kwargs.pop('prepare_model_fn', None)
            prepare_model_fn = kwargs.pop('prepare_model_fn', prepare_model_fn)
            self.model = self.model if prepare_model_fn is None else prepare_model_fn(self.model, **kwargs)

    def get_optim_kwargs(self, optim:Optimizer) -> Dict[str, Any]:
        opt_kwargs, _opt_kwargs = {}, {}
        if self.train_args.opt_kwargs is not None:
            for item in self.train_args.opt_kwargs.replace(" ", "").split(","):
                key, value = item.split("=")
                opt_kwargs[key] = value
        opt_kwargs.update(self.kwargs)
        if self.state.lr is not None:
            _opt_kwargs['lr'] = self.state.lr
        if self.state.weight_decay is not None:
            _opt_kwargs['weight_decay'] = self.state.weight_decay
        import ast
        import inspect
        signature = inspect.signature(optim.__init__)
        for name, param in signature.parameters.items():
            if name not in ("self", "params"):
                _opt_kwargs[name] = _opt_kwargs.get(name, 
                                                    self.train_args.get(name, 
                                                                        opt_kwargs.get(name, param.default)))
                if get_origin(param.annotation) not in [Union, tuple, list, dict]:
                    if not isinstance(_opt_kwargs[name], param.annotation):
                        _opt_kwargs[name] = ast.literal_eval(_opt_kwargs[name])
                else:
                    if isinstance(_opt_kwargs[name], str):
                        _opt_kwargs[name] = ast.literal_eval(_opt_kwargs[name])
        return _opt_kwargs
        
    def prepare_optimizer(self, **kwargs) -> None:
        if self.optimizer is None:
            if self.accelerator.is_main_process:
                logger.info(f"Optimizer initialize: {self.state.optim.upper()}")
            prepare_optimizer_fn = kwargs.pop('prepare_optimizer_fn', None)
            prepare_optimizer_fn = self.kwargs.pop('prepare_optimizer_fn', prepare_optimizer_fn)
            optim_cls = OPTIMIZERS[self.state.optim]
            self.optimizer = optim_cls(
                params=self.model.parameters(),
                **self.get_optim_kwargs(optim_cls),
                ) if prepare_optimizer_fn is None else prepare_optimizer_fn(
                    optim_cls,
                    model=self.model,
                    **self.get_optim_kwargs(optim_cls),
                )
            
    def prepare_lr_scheduler(self, **kwargs)->None:
        if self.lr_scheduler is None:
            if self.accelerator.is_main_process:
                logger.info(f"LR scheduler initialize: {self.state.lr_scheduler.upper()}")
            prepare_lr_scheduler_fn:Callable = kwargs.pop("prepare_lr_scheduler_fn", None)
            prepare_lr_scheduler_fn:Callable = self.kwargs.pop("prepare_lr_scheduler_fn", None) if prepare_lr_scheduler_fn is None else prepare_lr_scheduler_fn
            import ast
            lr_scheduler_kwargs = ast.literal_eval(self.train_args.lr_scheduler_kwargs) if isinstance(self.train_args.lr_scheduler_kwargs, str) else kwargs.pop("lr_scheduler_kwargs", self.train_args.lr_scheduler_kwargs)
            lr_scheduler_kwargs = self.kwargs.pop("lr_scheduler_kwargs", lr_scheduler_kwargs)
            self.lr_scheduler = get_scheduler(
                name=self.state.lr_scheduler,
                optimizer=self.optimizer,
                num_warmup_steps=self.state.warmup_steps,
                num_training_steps=self.state.max_steps,
                scheduler_specific_kwargs=lr_scheduler_kwargs,
            ) if prepare_lr_scheduler_fn is None else prepare_lr_scheduler_fn(
                name=self.state.lr_scheduler,
                optimizer=self.optimizer,
                num_warmup_steps=self.state.warmup_steps,
                num_training_steps=self.state.max_steps,
                scheduler_specific_kwargs=lr_scheduler_kwargs,
                **kwargs,
            )

    def have_accelerator_prepared(self)->None:
        if self.eval_dataloaders is not None:
            self.model, self.optimizer, self.lr_scheduler, self.train_dataloader, self.eval_dataloaders = self.accelerator.prepare(self.model, self.optimizer, self.lr_scheduler, self.train_dataloader, self.eval_dataloaders)
        else:
            self.model, self.optimizer, self.lr_scheduler, self.train_dataloader = self.accelerator.prepare(self.model, self.optimizer, self.lr_scheduler, self.train_dataloader)

    def train(self, **kwargs):
        self.get_train_dataloader()
        self.get_eval_dataloaders()
        self.calibrate_train_state()

        self.prepare_model(**kwargs)
        self.prepare_optimizer(**kwargs)
        self.prepare_lr_scheduler(**kwargs)

        self.have_accelerator_prepared()

        logger.info(" ***** Running training *****  ")
        logger.info(f" Num examples = {self.state.num_examples:,}")
        logger.info(f" Num Epochs = {self.state.epochs:,}")
        logger.info(f" Instantaneous batch size per device = {self.train_args.per_device_train_batch_size:,}")
        if self.train_args.per_device_train_batch_size != self.state.train_batch_size:
            logger.info(f" Training with DataParallel so batch size has been adjusted to: {self.state.train_batch_size:,}")
        logger.info(f" Total train batch size (w. parallel, distributed & accumulation) = {self.state.total_train_batch_size:,}")
        logger.info(f" Gradient Accumulation steps = {self.train_args.gradient_accumulation_steps}")
        logger.info(f" Total optimization steps = {self.state.max_steps:,}")
        logger.info(f" Number of trainable parameters = {get_model_param_count(self.model, trainable_only=True):,}")

        self.callback_handler.on_train_begin(
            state=self.state, 
            tb_writer=self.accelerator.get_tracker(str(LoggerType.TENSORBOARD))
            if self.train_args.tensorboard_project is not None else None,
        )
        
        execute_train_process:Callable = kwargs.pop("execute_train_process", self.kwargs.get("execute_train_process", None))
        execute_train_process = self.execute_train_process(
            **kwargs,
        ) if execute_train_process is None else execute_train_process(
            self.accelerator, 
            self.model, 
            self.optimizer, 
            self.lr_scheduler, 
            self.train_dataloader, 
            self.eval_dataloaders,
            self.state,
            self.callback_handler,
            **kwargs.update(self.kwargs),
        )

        self.accelerator.end_training()
        self.callback_handler.on_train_end(state=self.state)

    @contextmanager
    def execute_train_contexts(self, *models):
        if self.train_args.do_debug:
            with torch.autograd.set_detect_anomaly(True):
                with self.accelerator.accumulate(models):
                    yield
        else:
            with self.accelerator.accumulate(models):
                yield

    def execute_train_process(self, **kwargs):
        resume_step = self.resume_from_checkpoint()

        compute_loss:Callable = kwargs.pop("compute_loss", self.kwargs.get("compute_loss", self.compute_loss))
        for epoch in range(self.state.global_epoch, self.state.epochs):
            self.callback_handler.on_epoch_begin(state=self.state, **kwargs)

            epoch_iterator = self.train_dataloader
            if hasattr(epoch_iterator, "dataset") and isinstance(epoch_iterator.dataset, IterableDataset):
                epoch_iterator.set_epoch(epoch)
            if resume_step > 0:
                epoch_iterator = skip_first_batches(epoch_iterator, resume_step)
                resume_step = 0

            for _, batch in enumerate(epoch_iterator):
                self.callback_handler.on_step_begin(state=self.state, sync_on=self.accelerator.sync_gradients, **kwargs)
                
                with self.execute_train_contexts(self.model):
                    loss, metrics = compute_loss(self.model, batch, state=self.state, **kwargs)
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.unscale_gradients(self.optimizer)
                        if self.state.max_grad_norm > 0:
                            grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.state.max_grad_norm)
                            if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                                grad_norm = self.model.get_global_grad_norm()
                                if hasattr(grad_norm, "item"):
                                    grad_norm = grad_norm.item()
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                self.callback_handler.on_step_end(state=self.state, sync_on=self.accelerator.sync_gradients, **kwargs)

                if self.accelerator.sync_gradients:
                    self.model.zero_grad()
                    self.do_log(loss=loss, grad_norm=grad_norm, metrics=metrics, **kwargs)
                    self.do_evaluate(**kwargs)
                    self.do_save(**kwargs)
                
                if self.state.should_stop and not self.accelerator.sync_gradients:
                    self.accelerator.gradient_state._set_sync_gradients(True)
                    break
            
            self.callback_handler.on_epoch_end(state=self.state, **kwargs)
            if self.state.should_stop:
                break

    def prepare_inputs(self, data: Union[torch.Tensor, Any], **kwargs) -> Union[torch.Tensor, Any]:
        if isinstance(data, Mapping):
            inputs_kwargs = kwargs.pop("inputs_kwargs", None)
            if inputs_kwargs is None:
                return type(data)({k: self.prepare_inputs(v) for k, v in data.items()})
            else:
                inputs = {k: self.prepare_inputs(v) for k, v in data.items()}
                inputs.update(inputs_kwargs)
                return type(data)(inputs)
        elif isinstance(data, (tuple, list)):
            return type(data)(self.prepare_inputs(v) for v in data)
        elif isinstance(data, torch.Tensor):
            return data.to(**{"device": self.accelerator.device})
        return data

    def compute_loss(self, model:AutoModel, inputs: Dict[str, Any], **kwargs) -> Tuple[torch.Tensor, Dict[str, Any]]:
        prepare_inputs = kwargs.pop("prepare_inputs_fn", None)
        prepare_inputs = self.kwargs.get("prepare_inputs_fn", self.prepare_inputs)
        inputs.update(kwargs)
        inputs = prepare_inputs(inputs, **self.kwargs)
        outputs = model(**inputs)
        loss = outputs["loss"] if isinstance(outputs, Dict) else outputs[0]
        return (loss, None)

    def do_evaluate(self, **kwargs):
        if self.state.should_eval and self.eval_dataloaders is not None:
            if self.eval_dataloaders is None:
                self.get_eval_dataloader()
                if self.train_args.do_train:
                    if isinstance(self.eval_dataloaders, Dict):
                        for key, value in self.eval_dataloaders.items():
                            self.eval_dataloaders[key] = self.accelerator.prepare(value)
                    else:
                        self.eval_dataloaders = self.accelerator.prepare(self.eval_dataloaders)

            exec_eval_fn:Callable = kwargs.get("exec_eval_fn", None)
            if isinstance(self.eval_dataloaders, Dict):
                for key, eval_dataloader in self.eval_dataloaders.items():
                    if exec_eval_fn is None:
                        self.do_log(**self.execute_eval_process(eval_dataloader, **kwargs))
                    else:
                        exec_eval_params = {
                            "model": self.model,
                            "eval_dataloader": eval_dataloader,
                            "state": self.state,
                            "description": key.upper(),
                        }
                        kwargs.update(exec_eval_params)
                        exec_eval_fn(**kwargs)
            else:
                if exec_eval_fn is None:
                    self.do_log(prefix='eval', **self.execute_eval_process(self.eval_dataloaders, **kwargs))
                else:
                    exec_eval_params = {
                        "model": self.model,
                        "eval_dataloader": self.eval_dataloaders,
                        "state": self.state,
                    }
                    kwargs.update(exec_eval_params)
                    self.do_log(**exec_eval_fn(**kwargs))

    @torch.inference_mode()
    def execute_eval_process(self, eval_dataloader, **kwargs) -> Dict[str, Any]:        

        execute_metrics:Callable = kwargs.pop("execute_metrics", self.compute_loss)
        logger.info(f" **** RUNNING {kwargs.get('description', 'EVALUATION')} ****", extra={"prefix":"\n\r"})
        logger.info(f" NUM EXAMPLES = {len(eval_dataloader)}")
        logger.info(f" BATCH SIZE = {self.train_args.per_device_eval_batch_size * self.accelerator.num_processes}")
        self.callback_handler.on_eval_begin(state=self.state, eval_dataloader=eval_dataloader, **kwargs)

        self.model.eval()
        _metrics = {"loss": [], "metrics": {}}
        for _, batch in enumerate(eval_dataloader):
            loss, metrics = execute_metrics(self.model, batch, **kwargs)
            _metrics["loss"].append(torch.atleast_1d(loss))
            if metrics is not None:
                for key in metrics:
                    if key in _metrics["metrics"]:
                        _metrics["metrics"].append(metrics[key])
                    else:
                        _metrics["metrics"] = [metrics[key]]
            self.callback_handler.on_eval_step(state=self.state, **kwargs)

        _metrics["loss"] = torch.cat(_metrics["loss"], dim=-1)
        for key, item in _metrics["metrics"]:
            _metrics["metrics"][key] = torch.cat(item, dim=-1) if isinstance(item, torch.Tensor) else item
        _metrics = self.accelerator.gather_for_metrics(_metrics)
        _metrics["loss"] = _metrics["loss"].mean().item()
        for key, item in _metrics["metrics"]:
            _metrics["metrics"][key] = item.mean().item() if isinstance(item, torch.Tensor) else np.mean(item)
        
        self.callback_handler.on_eval_end(state=self.state)

        return _metrics

    def do_save(self, **kwargs):
        if self.state.should_save:
            output_dir = os.path.join(
                self.train_args.project,
                f"checkpoint-{self.state.global_epoch}-{self.state.global_step}"
            )
            if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                self.accelerator.unwrap_model(self.model).save_pretrained(
                    output_dir, 
                    state_dict=self.accelerator.get_state_dict(self.model),
                    safe_serialization=True,
                )
                self.model.save_checkpoint(output_dir)
            else:
                self.accelerator.save_state(output_dir)
            rotate_checkpoints(self.train_args.save_total_limit, self.train_args.project)
            self.callback_handler.on_save(state=self.state, **kwargs)

    def do_log(self, loss, grad_norm = None, metrics:Dict[str, Any] = None, prefix:str = None, **kwargs):
        if self.state.should_log:
            metrics = {} if metrics is None else metrics
            if isinstance(loss, torch.Tensor):
                loss = loss.detach()
                loss = self.accelerator.gather(loss).mean().item()
            else:
                loss = np.mean(self.accelerator.gather_for_metrics([loss]))
            metrics["epoch"] = (self.state.global_step / self.state.max_steps) * self.state.epochs
            metrics["loss"] = round(loss, 4)
            if grad_norm is not None:
                metrics["grad_norm"] = grad_norm.detach().item() if isinstance(grad_norm, torch.Tensor) else grad_norm
                metrics["lr"] = round(self.lr_scheduler.get_last_lr()[0], 8)
            if prefix is not None:
                _metrics = {}
                for k, v in metrics.items():
                    _metrics[f"{prefix}_{k}"] = v
                metrics = _metrics
            self.callback_handler.on_log(state=self.state, logs=metrics, **kwargs)

    def resume_from_checkpoint(self) -> int:
        if self.train_args.resume_from_checkpoint is not None:
            logger.info(f"Resume from checkpoint: {self.train_args.resume_from_checkpoint.split('/')[0]}")
            self.accelerator.load_state(self.train_args.resume_from_checkpoint)
            import re
            match = re.search(r"checkpoint-(\d+)-(\d+)", self.train_args.resume_from_checkpoint)
            self.state.global_epoch = int(match.group(1))
            self.state.global_step = int(match.group(2))
            logger.info(f"Continuing training from epoch {self.state.global_epoch}")
            logger.info(f"Continuing training from step {self.state.global_step}")
            return self.state.global_step * self.train_args.gradient_accumulation_steps - self.state.global_epoch * len(self.train_dataloader)
        return 0