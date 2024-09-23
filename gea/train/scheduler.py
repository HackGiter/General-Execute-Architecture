from typing import Optional, Union, List
from functools import partial

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from transformers import SchedulerType, get_scheduler
from transformers.trainer_pt_utils import LayerWiseDummyOptimizer, LayerWiseDummyScheduler
from transformers import (
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_inverse_sqrt_schedule,
    get_wsd_schedule
)
from transformers.optimization import (
    get_reduce_on_plateau_schedule,
    _get_cosine_schedule_with_warmup_lr_lambda,
)

def get_cosine_with_min_lr_schedules_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: Union[int, List[int]],
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
    min_lr: Union[float, List[float]] = None,
    min_lr_rate: Union[float, List[float]] = None,   
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to min_lr, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
        min_lr (`float`, *optional*):
            The minimum learning rate to reach after the cosine schedule.
        min_lr_rate (`float`, *optional*):
            The minimum learning rate as a ratio of the initial learning rate. If set, `min_lr` should not be set.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    if min_lr is not None:
        min_lr = min_lr if isinstance(min_lr, List) else [min_lr]
        min_lr_rate = [item / optimizer.defaults["lr"] for item in min_lr]
    
    num_warmup_steps = num_warmup_steps if isinstance(num_warmup_steps, List) else [num_warmup_steps]
    min_lr_rate = min_lr_rate if isinstance(min_lr_rate, List) else [min_lr_rate]

    if len(num_warmup_steps) != len(min_lr_rate):
        raise ValueError("Length of min_lr_rate should be equal to length of num_warmup_steps")

    lr_lambdas = [
        partial(
            _get_cosine_schedule_with_warmup_lr_lambda,
            num_warmup_steps=it1,
            num_training_steps=num_training_steps,
            num_cycles=num_cycles,
            min_lr_rate=it2,
        )
        for it1, it2 in zip(num_warmup_steps, min_lr_rate)
    ]
    if len(optimizer.param_groups) != len(lr_lambdas):
        if len(lr_lambdas) == 1:
            lr_lambdas = lr_lambdas * len(optimizer.param_groups)
        else:
            raise ValueError("min_lr_rate should be float or list of one element if size of optimizer.param_groups is larger than size of min_lr_rate")
    return LambdaLR(optimizer, lr_lambdas, last_epoch)

TYPE_TO_SCHEDULER_FUNCTION = {
    SchedulerType.LINEAR: get_linear_schedule_with_warmup,
    SchedulerType.COSINE: get_cosine_schedule_with_warmup,
    SchedulerType.COSINE_WITH_RESTARTS: get_cosine_with_hard_restarts_schedule_with_warmup,
    SchedulerType.POLYNOMIAL: get_polynomial_decay_schedule_with_warmup,
    SchedulerType.CONSTANT: get_constant_schedule,
    SchedulerType.CONSTANT_WITH_WARMUP: get_constant_schedule_with_warmup,
    SchedulerType.INVERSE_SQRT: get_inverse_sqrt_schedule,
    SchedulerType.REDUCE_ON_PLATEAU: get_reduce_on_plateau_schedule,
    SchedulerType.COSINE_WITH_MIN_LR: get_cosine_with_min_lr_schedules_with_warmup,
    SchedulerType.WARMUP_STABLE_DECAY: get_wsd_schedule,
}


def get_schedulers(
    name: Union[str, SchedulerType],
    optimizer: Optimizer,
    num_warmup_steps: Optional[int] = None,
    num_training_steps: Optional[int] = None,
    scheduler_specific_kwargs: Optional[dict] = None,
):
    """
    Unified API to get any scheduler from its name.

    Args:
        name (`str` or `SchedulerType`):
            The name of the scheduler to use.
        optimizer (`torch.optim.Optimizer`):
            The optimizer that will be used during training.
        num_warmup_steps (`int`, *optional*):
            The number of warmup steps to do. This is not required by all schedulers (hence the argument being
            optional), the function will raise an error if it's unset and the scheduler type requires it.
        num_training_steps (`int``, *optional*):
            The number of training steps to do. This is not required by all schedulers (hence the argument being
            optional), the function will raise an error if it's unset and the scheduler type requires it.
        scheduler_specific_kwargs (`dict`, *optional*):
            Extra parameters for schedulers such as cosine with restarts. Mismatched scheduler types and scheduler
            parameters will cause the scheduler function to raise a TypeError.
    """
    name = SchedulerType(name)
    schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]

    # If a `LayerWiseDummyOptimizer` is passed we extract the optimizer dict and
    # recursively call `get_scheduler` to get the proper schedulers on each parameter
    if optimizer is not None and isinstance(optimizer, LayerWiseDummyOptimizer):
        optimizer_dict = optimizer.optimizer_dict
        scheduler_dict = {}

        for param in optimizer_dict.keys():
            scheduler_dict[param] = get_scheduler(
                name,
                optimizer=optimizer_dict[param],
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )

        def scheduler_hook(param):
            # Since the optimizer hook has been already attached we only need to
            # attach the scheduler hook, the gradients have been zeroed here
            scheduler_dict[param].step()

        for param in optimizer_dict.keys():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(scheduler_hook)

        return LayerWiseDummyScheduler()

    if name == SchedulerType.CONSTANT:
        return schedule_func(optimizer)

    if scheduler_specific_kwargs is None:
        scheduler_specific_kwargs = {}

    if name == SchedulerType.REDUCE_ON_PLATEAU:
        return schedule_func(optimizer, **scheduler_specific_kwargs)

    # All other schedulers require `num_warmup_steps`
    if num_warmup_steps is None:
        raise ValueError(f"{name} requires `num_warmup_steps`, please provide that argument.")

    if name == SchedulerType.CONSTANT_WITH_WARMUP:
        return schedule_func(optimizer, num_warmup_steps=num_warmup_steps)

    if name == SchedulerType.INVERSE_SQRT:
        return schedule_func(optimizer, num_warmup_steps=num_warmup_steps)

    if name == SchedulerType.WARMUP_STABLE_DECAY:
        return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, **scheduler_specific_kwargs)

    # All other schedulers require `num_training_steps`
    if num_training_steps is None:
        raise ValueError(f"{name} requires `num_training_steps`, please provide that argument.")

    return schedule_func(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        **scheduler_specific_kwargs,
    )
