import os
import math
import random
from contextlib import contextmanager
from typing import Callable, Mapping, Union, Tuple, Dict, Any, get_origin

import numpy as np

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, RandomSampler

from datasets import Dataset, IterableDataset, DatasetDict
from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    get_scheduler,
)
from transformers.data.data_collator import DataCollatorForSeq2Seq
from transformers.trainer_utils import TrainerMemoryTracker
from accelerate import Accelerator, DataLoaderConfiguration, skip_first_batches
from accelerate.utils import (
    LoggerType, 
    DistributedType, 
    GradientAccumulationPlugin,
)

from  ..args import TrainArguments
from ..utils.callback import CallbackHandler, TrainStateCallback, TrainState, StateStrategy
from ..utils.integration import TensorBoardCallback
from ..utils.tools import rotate_checkpoints, get_decay_parameter_names
from ..utils.logging import get_logger
from ..utils.constant import IGNORE_INDEX, OPTIMIZERS, ALL_LAYERNORM_LAYERS

# from .scheduler import get_schedulers
from .checkpointing import upcast_layernorm, upcast_lmhead_output, gradient_checkpoint

from transformers.trainer_pt_utils import (
    metrics_format, 
    get_model_param_count, 
    remove_dummy_checkpoint,
    distributed_concat,
)

logger = get_logger(__name__)

class Trainer:
    """
    Scheduler for training including optimizer, lr scheduler, loss calculation and etc
    """
    def __init__(
            self,
            model: Union[AutoModel, PreTrainedModel, nn.Module],
            tokenizer: AutoTokenizer,
            train_args: TrainArguments,
            train_dataset: Dataset = None,
            eval_dataset: Union[Dataset, DatasetDict] = None,
            **kwargs,
            ) -> None:
        self.train_args = train_args
        self.model = model
        self.model_ = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_datasets = eval_dataset

        self.accelerator = Accelerator(
            log_with=LoggerType.TENSORBOARD if self.train_args.tensorboard_project is not None else LoggerType.WANDB,
            project_dir=self.train_args.project,
            **self.prepare_accelerator_kwargs()
        )
        if self.train_args.tensorboard_project is not None:
            logger.info("Tensoboard tracker initialize")
            self.accelerator.init_trackers(project_name=self.train_args.tensorboard_project)
        if self.train_args.track_memory_usage:
            self.memory_tracker = TrainerMemoryTracker()
            self.memory_tracker.start()
        else:
            self.memory_tracker = None
        self.accelerator.free_memory()

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
            gradient_accumulation_steps=self.train_args.gradient_accumulation_steps,
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

    def prepare_accelerator_kwargs(self) -> Dict[str, Any]:
        dataloader_config = DataLoaderConfiguration(
            dispatch_batches=isinstance(self.train_dataset, IterableDataset),
            even_batches=self.train_args.even_batches,
            use_seedable_sampler=self.train_args.use_seedable_sampler,
            non_blocking=self.train_args.non_blocking
        )
        gradient_accumulation_plugin = GradientAccumulationPlugin(
            num_steps=self.train_args.gradient_accumulation_steps,
            adjust_scheduler=True,
            sync_with_dataloader=False,
            sync_each_batch=False,
        )

        return {
            "mixed_precision": self.train_args.mixed_precision,
            "dataloader_config": dataloader_config,
            "gradient_accumulation_plugin": gradient_accumulation_plugin,
        }

    def prepare_train_kwargs(self, kwargs:Dict[str, Any]) -> None:
        self.train_data_collator = kwargs.pop("train_collate_fn", 
                                              DataCollatorForSeq2Seq(
                                                  tokenizer=self.tokenizer, 
                                                  pad_to_multiple_of=8, 
                                                  label_pad_token_id=kwargs.pop("ignore_index", IGNORE_INDEX)))
        self.eval_data_collator = kwargs.pop("eval_collate_fn", self.train_data_collator)

        self.train_dataloader = None
        self.eval_dataloaders = None
        self.train_sampler = kwargs.pop("train_sampler", RandomSampler(self.train_dataset) if self.train_args.shuffle else None)
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
                "shuffle": self.train_args.shuffle if self.train_sampler is None else None,
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
            if self.train_args.upcast_layernorm:
                upcast_layernorm(self.model)
            if self.train_args.gradient_checkpointing:
                gradient_checkpoint(self.model)
            if self.train_args.upcast_lmhead_output:
                upcast_lmhead_output(self.model)

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
            logger.info(f"Optimizer initialize: {self.state.optim.upper()}")
            # decay_parameters = get_parameter_names(self.model, ALL_LAYERNORM_LAYERS)
            decay_parameters = get_decay_parameter_names(self.model, ALL_LAYERNORM_LAYERS)
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in self.model.named_parameters() if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.train_args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in self.model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]
            prepare_optimizer_fn = kwargs.pop('prepare_optimizer_fn', None)
            prepare_optimizer_fn = self.kwargs.pop('prepare_optimizer_fn', prepare_optimizer_fn)
            optim_cls = OPTIMIZERS[self.state.optim]
            self.optimizer = optim_cls(
                params=optimizer_grouped_parameters,
                **self.get_optim_kwargs(optim_cls),
                ) if prepare_optimizer_fn is None else prepare_optimizer_fn(
                    optim_cls,
                    model=self.model,
                    **self.get_optim_kwargs(optim_cls),
                )
            
    def prepare_lr_scheduler(self, **kwargs)->None:
        if self.lr_scheduler is None:
            logger.info(f"LR scheduler initialize: {self.state.lr_scheduler.upper()}")
            prepare_lr_scheduler_fn:Callable = kwargs.pop("prepare_lr_scheduler_fn", None)
            prepare_lr_scheduler_fn:Callable = self.kwargs.pop("prepare_lr_scheduler_fn", None) if prepare_lr_scheduler_fn is None else prepare_lr_scheduler_fn
            import ast
            lr_scheduler_kwargs = ast.literal_eval(self.train_args.lr_scheduler_kwargs) if isinstance(self.train_args.lr_scheduler_kwargs, str) else kwargs.pop("lr_scheduler_kwargs", self.train_args.lr_scheduler_kwargs)
            lr_scheduler_kwargs = self.kwargs.pop("lr_scheduler_kwargs", lr_scheduler_kwargs)
            self.lr_scheduler = get_scheduler(
                name=self.state.lr_scheduler,
                optimizer=self.optimizer,
                # num_warmup_steps=self.state.warmup_steps * self.accelerator.num_processes,
                # num_training_steps=self.state.max_steps * self.accelerator.num_processes,
                num_warmup_steps=self.state.warmup_steps,
                num_training_steps=self.state.max_steps,
                scheduler_specific_kwargs=lr_scheduler_kwargs,
            ) if prepare_lr_scheduler_fn is None else prepare_lr_scheduler_fn(
                name=self.state.lr_scheduler,
                optimizer=self.optimizer,
                # num_warmup_steps=self.state.warmup_steps * self.accelerator.num_processes,
                # num_training_steps=self.state.max_steps * self.accelerator.num_processes,
                num_warmup_steps=self.state.warmup_steps,
                num_training_steps=self.state.max_steps,
                scheduler_specific_kwargs=lr_scheduler_kwargs,
                **kwargs,
            )

    def prepare_for_training(self) -> None:
        if hasattr(self.lr_scheduler, "step"):
            self.model, self.optimizer, self.train_dataloader = self.accelerator.prepare(self.model, self.optimizer, self.train_dataloader)
        else:
            self.model, self.optimizer, self.lr_scheduler, self.train_dataloader = self.accelerator.prepare(self.model, self.optimizer, self.lr_scheduler, self.train_dataloader)
        if self.eval_dataloaders is not None:
            self.eval_dataloaders = self.accelerator.prepare(self.eval_dataloaders)
        self.model.zero_grad()

    def train(self, **kwargs):
        self.get_train_dataloader()
        self.get_eval_dataloaders()
        self.calibrate_train_state()

        self.prepare_model(**kwargs)
        self.prepare_optimizer(**kwargs)
        self.prepare_lr_scheduler(**kwargs)
        self.prepare_for_training()

        logger.info(" ***** Running training *****  ")
        logger.info(f" Num examples = {self.state.num_examples:,}")
        logger.info(f" Num Epochs = {self.state.epochs:,}")
        logger.info(f" Instantaneous batch size per device = {self.train_args.per_device_train_batch_size:,}")
        if self.train_args.per_device_train_batch_size != self.state.train_batch_size:
            logger.info(f" Training with DataParallel so batch size has been adjusted to: {self.state.train_batch_size:,}")
        logger.info(f" Total train batch size (w. parallel, distributed & accumulation) = {self.state.train_batch_size * self.state.gradient_accumulation_steps:,}")
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

        train_metrics = metrics_format(self, self.state.get_train_metric("train"))
        k_width = max(len(str(x)) for x in train_metrics.keys())
        v_width = max(len(str(x)) for x in train_metrics.values())
        train_metrics_info = ""
        for key in sorted(train_metrics.keys()):
            train_metrics_info += f"  {key: <{k_width}} = {train_metrics[key]:>{v_width}}\n"
        logger.info("\n***** train metrics *****\n" + train_metrics_info)

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

        resume_step_in_current_epoch = self.resume_from_checkpoint()
        
        sync_gradient = False
        cur_loss = torch.tensor(0.0).to(self.accelerator.device)
        compute_loss:Callable = kwargs.pop("compute_loss", self.kwargs.get("compute_loss", self.compute_loss))

        for epoch in range(self.state.global_epoch, self.state.epochs):
            self.callback_handler.on_epoch_begin(state=self.state, **kwargs)

            epoch_iterator = self.train_dataloader
            if hasattr(epoch_iterator, "dataset"):
                epoch_iterator.set_epoch(epoch)
            epoch_iterator = skip_first_batches(epoch_iterator, resume_step_in_current_epoch) if resume_step_in_current_epoch > 0 else epoch_iterator
            resume_step_in_current_epoch = 0

            for _, batch in enumerate(epoch_iterator):
                # self.callback_handler.on_step_begin(state=self.state, sync_on=self.accelerator.sync_gradients, **kwargs)
                sync_gradient = self.callback_handler.on_step_begin(state=self.state, sync_on=sync_gradient, **kwargs)
                
                with self.execute_train_contexts(self.model):
                    loss, metrics = compute_loss(model=self.model, inputs=batch, **kwargs)
                    self.accelerator.backward(loss)
                cur_loss += loss.detach() / self.state.gradient_accumulation_steps
                # if self.accelerator.sync_gradients:
                if sync_gradient:
                    # self.accelerator.unscale_gradients(self.optimizer)
                    if self.state.max_grad_norm > 0:
                        _grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.state.max_grad_norm)
                        if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                            grad_norm = self.model.get_global_grad_norm()
                            if hasattr(grad_norm, "item"):
                                grad_norm = grad_norm.item()
                        else:
                            grad_norm = _grad_norm
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.state.max_grad_norm)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.model.zero_grad()

                self.callback_handler.on_step_end(
                    state=self.state, 
                    # flops=float(self.calculate_floating_point_ops(batch)), 
                    flops=None,
                    # sync_on=self.accelerator.sync_gradients, 
                    sync_on=sync_gradient,
                    **kwargs)

                # if self.accelerator.sync_gradients:
                if sync_gradient:
                    self.do_log(
                        loss=cur_loss, 
                        # loss=None,
                        grad_norm=grad_norm, 
                        # grad_norm=None,
                        flops=self.state.cur_flops,
                        metrics=metrics, 
                        **kwargs)
                    self.do_evaluate(**kwargs)
                    self.do_save(**kwargs)
                
                if self.state.should_stop:
                    if not self.accelerator.sync_gradients:
                        self.accelerator.gradient_state._set_sync_gradients(True)
                    break

            self.callback_handler.on_epoch_end(state=self.state, **kwargs)
            if self.state.should_stop:
                self.do_save(**kwargs)
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
            kwargs = {"device": self.accelerator.device}
            if self.accelerator.distributed_type == DistributedType.DEEPSPEED and (torch.is_floating_point(data) or torch.is_complex(data)):
                # NLP models inputs are int/uint and those get adjusted to the right dtype of the
                # embedding. Other models such as wav2vec2's inputs are already float and thus
                # may need special handling to match the dtypes of the model
                kwargs.update({"dtype": self.accelerator.state.deepspeed_plugin.hf_ds_config.dtype()})
            return data.to(**kwargs)
        return data

    def compute_loss(self, model:AutoModel, inputs: Dict[str, Any], **kwargs) -> Tuple[torch.Tensor, Dict[str, Any]]:
        prepare_inputs = kwargs.pop("prepare_inputs_fn", None)
        prepare_inputs = self.kwargs.get("prepare_inputs_fn", self.prepare_inputs)
        inputs.update(kwargs)
        inputs = prepare_inputs(inputs, **self.kwargs)
        outputs = model(**inputs)
        loss = outputs["loss"] if isinstance(outputs, Dict) else outputs[0]
        del inputs
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
                        self.do_log(prefix='eval', **self.execute_eval_process(eval_dataloader, **kwargs))
                    else:
                        exec_eval_params = {
                            "model": self.model,
                            "eval_dataloader": eval_dataloader,
                            "state": self.state,
                            "description": key.upper(),
                        }
                        kwargs.update(exec_eval_params)
                        self.do_log(prefix='eval', **exec_eval_fn(**kwargs))
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
                    self.do_log(prefix='eval', **exec_eval_fn(**kwargs))
            torch.cuda.empty_cache()

    @torch.inference_mode()
    def execute_eval_process(self, eval_dataloader, **kwargs) -> Dict[str, Any]:        

        execute_metrics:Callable = kwargs.pop("execute_metrics", self.kwargs.get("execute_metrics", self.compute_loss))
        logger.info(f" **** RUNNING {kwargs.pop('description', 'EVALUATION')} ****", extra={"prefix":"\n\r"})
        logger.info(f" NUM EXAMPLES = {len(eval_dataloader)}")
        logger.info(f" BATCH SIZE = {self.train_args.per_device_eval_batch_size * self.accelerator.num_processes}")
        self.callback_handler.on_eval_begin(state=self.state, eval_dataloader=eval_dataloader, **kwargs)

        self.model.eval()
        _metrics = {"loss": [], "metrics": {}}
        for _, batch in enumerate(eval_dataloader):
            loss, metrics = execute_metrics(self.model, batch, state=self.state, **kwargs)
            _metrics["loss"].append(torch.atleast_1d(loss))
            if metrics is not None:
                for k, v in metrics.items():
                    v = v if isinstance(v, torch.Tensor) else torch.stack(v, dim=-1)
                    if k in _metrics["metrics"]:
                        _metrics["metrics"][k].append(v)
                    else:
                        _metrics["metrics"][k] = [v]
            self.callback_handler.on_eval_step(state=self.state, **kwargs)

        _metrics["loss"] = torch.cat(_metrics["loss"], dim=-1)
        for k, v in _metrics["metrics"].items():
            v = torch.stack(v, 0).mean(0)
            _metrics["metrics"][k] = v if v.dim() == 0 else list(torch.unbind(v))
        self.callback_handler.on_eval_end(state=self.state)

        return _metrics

    def do_save(self, **kwargs):
        if self.state.should_save:
            logger.info(f" **** SAVING checkpoint-{self.state.global_epoch}-{self.state.global_step} in {self.train_args.project} ****", extra={"prefix":"\n\r"})
            output_dir = os.path.join(
                self.train_args.project,
                f"checkpoint-{self.state.global_epoch}-{self.state.global_step}"
            )
            self.accelerator.save_state(output_dir)
            self.state.save_to_json(os.path.join(self.train_args.project, 'train_state.json'))
            
            if self.accelerator.distributed_type == DistributedType.DEEPSPEED and self.state.should_stop:
                
                if self.accelerator.deepspeed_config["zero_optimization"]["stage"] == 3:
                    state_dict = self.accelerator.get_state_dict(self.model) if self.accelerator.deepspeed_config["zero_optimization"]['stage3_gather_16bit_weights_on_model_save'] else {}
                else:
                    state_dict = self.accelerator.get_state_dict(self.model)
                self.accelerator.unwrap_model(self.model).save_pretrained(
                    self.train_args.project, 
                    state_dict=state_dict,
                    is_main_process=self.accelerator.is_main_process,
                    save_function=self.accelerator.save,
                    safe_serialization=True,
                )
                if state_dict is not None and len(state_dict) == 0:
                    remove_dummy_checkpoint(self.accelerator.is_main_process, self.train_args.project, ['pytorch_model.bin', 'model.safetensors'])
                    self.model.save_checkpoint(self.train_args.project)

                if self.accelerator.is_main_process and self.tokenizer is not None:
                    self.tokenizer.save_pretrained(self.train_args.project)

            if self.state.is_world_process_zero:
                rotate_checkpoints(self.train_args.save_total_limit, self.train_args.project)
            
            self.callback_handler.on_save(state=self.state, **kwargs)

    def do_log(
            self, 
            loss:Union[torch.Tensor, Any] = None, 
            grad_norm:Union[torch.Tensor, Any] = None, 
            flops:float = None,
            metrics:Dict[str, Any] = None, 
            prefix:str = None, **kwargs):
        if self.state.should_log:
            if metrics is not None:
                _metrics = self.accelerator.gather_for_metrics(metrics)
                for k, v in _metrics.items():
                    v = v.mean(-1).item() if isinstance(v, torch.Tensor) else torch.stack(v, dim=0).mean(-1)
                    _metrics[k] = [round(_v, 6) for _v in v.cpu().numpy().tolist()] if isinstance(v, torch.Tensor) else round(v, 6)
            else:
                _metrics = {}

            metrics = {}
            # loss = self.accelerator.gather(loss).mean().item()
            if loss is not None and isinstance(loss, torch.Tensor):
                metrics["loss"] = round(self._nested_gather(loss).mean().item() / self.state.logging_steps, 4)
                loss -= loss

            if grad_norm is not None:
                metrics["grad_norm"] = round(grad_norm.detach().item(), 5) if isinstance(grad_norm, torch.Tensor) else round(grad_norm, 5)
                metrics["lr"] = round(self.lr_scheduler.get_last_lr()[0], 8)
            
            if flops is not None:
                metrics["flops"] = np.sum(self.accelerator.gather_for_metrics([flops])).item()
        
            metrics = {**metrics, **_metrics}
            if prefix is not None:
                _metrics = {}
                for k, v in metrics.items():
                    _metrics[f"{prefix}_{k}"] = v
                metrics = _metrics
            metrics["epoch"] = (self.state.global_step / self.state.max_steps) * self.state.epochs
            self.callback_handler.on_log(state=self.state, logs=metrics, **kwargs)

    def resume_from_checkpoint(self) -> int:
        if self.train_args.resume_from_checkpoint is not None:
            logger.info(f"Resume from checkpoint: {self.train_args.resume_from_checkpoint.split('/')[0]}")
            self.accelerator.load_state(self.train_args.resume_from_checkpoint)
            self.state = self.state.load_from_json(os.path.join(self.train_args.resume_from_checkpoint, 'train_state.json'))

            import re
            match = re.search(r"checkpoint-(\d+)-(\d+)", self.train_args.resume_from_checkpoint)
            self.state.global_epoch = int(match.group(1))
            self.state.global_step = int(match.group(2))
            logger.info(f"Resuming training from epoch {self.state.global_epoch}")
            logger.info(f"Resuming training from step {self.state.global_step}")

            return self.state.global_step * self.train_args.gradient_accumulation_steps - self.state.global_epoch * len(self.train_dataloader)
        return 0
    
    def save_rng_state(self, output_dir:str):
        rng_states = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "cpu": torch.random.get_rng_state(),
        }
        if self.accelerator.num_processes <= 1:
            rng_states["cuda"] = torch.cuda.random.get_rng_state()
            torch.save(rng_states, os.path.join(output_dir, "rng_state.pth"))
        else:
            rng_states['cuda'] = torch.cuda.random.get_rng_state_all()
            torch.save(rng_states, os.path.join(output_dir, f"rng_state_{self.accelerator.process_index}.pth"))

    def estimate_inputs(self, input_dict: Dict[str, Union[torch.Tensor, Any]]) -> int:
        numel = 0
        if not hasattr(self.model_, "warnings_issued"):
            self.model_.warnings_issued = {}
        if self.model_.main_input_name in input_dict:
            return input_dict[self.model_.main_input_name].numel()
        else:
            for _, v in input_dict.items():
                if isinstance(v, torch.Tensor):
                    numel += v.shape[0] * v.shape[1]
        return numel

    def _nested_gather(self, tensors):
        """
        Gather value of `tensors` (tensor or list/tuple of nested tensors) and convert them to numpy before
        concatenating them to `gathered`
        """
        if tensors is None:
            return

        tensors = distributed_concat(tensors)
        return tensors

    def calculate_floating_point_ops(
        self, input_dict: Dict[str, Union[torch.Tensor, Any]], exclude_embeddings: bool = True
    ) -> int:
        return 6 * self.estimate_inputs(input_dict) * self.model_.num_parameters(exclude_embeddings=exclude_embeddings)