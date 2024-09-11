import importlib
from typing import Iterable, Dict, List

from torch.utils import tensorboard

from .logging import get_logger
from .callback import StateCallback, TrainState

logger = get_logger(__name__)

def is_tensorboard_available():
    return importlib.util.find_spec("tensorboard") is not None or importlib.util.find_spec("tensorboardX") is not None

def rewrite_logs(d):
    new_d = {}
    train_prefix = "train_"
    train_prefix_len = len(train_prefix)
    eval_prefix = "eval_"
    eval_prefix_len = len(eval_prefix)
    test_prefix = "test_"
    test_prefix_len = len(test_prefix)
    for k, v in d.items():
        if k.startswith(eval_prefix):
            new_d["eval/" + k[eval_prefix_len:]] = v
        elif k.startswith(test_prefix):
            new_d["test/" + k[test_prefix_len:]] = v
        elif k.startswith(train_prefix):
            new_d["train/" + k[train_prefix_len:]] = v
        else:
            new_d["train/" + k] = v
    return new_d

class TensorBoardCallback(StateCallback):

    """
    A [`TrainerCallback`] that sends the logs to [TensorBoard](https://www.tensorflow.org/tensorboard).

    Args:
        tb_writer (`SummaryWriter`, *optional*):
            The writer to use. Will instantiate one if not set.
    """

    def __init__(self, tb_writer=None):
        has_tensorboard = is_tensorboard_available()
        if not has_tensorboard:
            raise RuntimeError(
                "TensorBoardCallback requires tensorboard to be installed. Either update your PyTorch version or"
                " install tensorboardX."
            )
        if has_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter  # noqa: F401

                self._SummaryWriter = SummaryWriter
            except ImportError:
                try:
                    from tensorboardX import SummaryWriter

                    self._SummaryWriter = SummaryWriter
                except ImportError:
                    self._SummaryWriter = None
        else:
            self._SummaryWriter = None
        self.tb_writer = tb_writer

    def _init_summary_writer(self, log_dir=None):
        if self._SummaryWriter is not None:
            self.tb_writer = self._SummaryWriter(log_dir=log_dir)

    def on_train_begin(self, state:TrainState, **kwargs):
        if not state.is_world_process_zero:
            return
        
        logger.info("Tensorboard tracker initialize")
        log_dir = kwargs.get("log_dir", None)
        _tb_writer = kwargs.pop("tb_writer", None)
        self.tb_writer:tensorboard.SummaryWriter = None if _tb_writer is None else _tb_writer.tracker

        if self.tb_writer is None:
            self._init_summary_writer(log_dir)

        if self.tb_writer is not None:
            import json
            if "model" in kwargs:
                model = kwargs["model"]
                if hasattr(model, "config") and model.config is not None:
                    model_config_json = model.config.to_json_string()
                    self.tb_writer.add_text("model_config", model_config_json)
            if "args" in kwargs:
                self.tb_writer.add_text("args", json.dumps(kwargs["args"]))
            if state is not None:
                self.tb_writer.add_hparams(state.hparams, metric_dict={})
                self.tb_writer.flush()

    def on_log(self, state:TrainState, logs=None, **kwargs):
        if not state.is_world_process_zero:
            return

        if self.tb_writer is None:
            self._init_summary_writer(kwargs.get("log_dir", None))

        if self.tb_writer is not None:
            logs = rewrite_logs(logs)
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    self.tb_writer.add_scalar(k, v, state.global_step)
                elif isinstance(v, Dict):
                    self.tb_writer.add_scalars(k, v, state.global_step)
                elif isinstance(v, (Iterable, List)):
                    self.tb_writer.add_scalars(k, { i:_v for i, _v in enumerate(v) }, state.global_step)
                else:
                    logger.warning(
                        "Trainer is attempting to log a value of "
                        f'"{v}" of type {type(v)} for key "{k}" as a scalar. '
                        "This invocation of Tensorboard's writer.add_scalar() "
                        "is incorrect so we dropped this attribute."
                    )
            self.tb_writer.flush()

    def on_train_end(self, **kwargs):
        if self.tb_writer:
            self.tb_writer.close()
            self.tb_writer = None