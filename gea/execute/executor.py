import ast
from typing import Dict, Any

from ..args.parser import parse_args

from ..model.template import get_model
from ..data.profile import get_dataset
from ..train.train import Trainer
from ..utils.tools import handle_unknown_kwargs

def run_train(kwargs):
    model_args, data_args, train_args, eval_args, unknown_args = parse_args()
    unknown_args:Dict[str, Any] = ast.literal_eval(handle_unknown_kwargs(unknown_args))
    kwargs.update(unknown_args)
    if kwargs.get("get_model_fn", None) is not None:
        model, tokenizer = kwargs.pop("get_model_fn")(model_args, **kwargs)
    else:
        model, tokenizer = get_model(model_args, **kwargs)
    if kwargs.get("get_dataset_fn", None) is not None:
        dataset = kwargs.pop("get_dataset_fn")(
            train_args=train_args, 
            model_args=model_args, 
            data_args=data_args, 
            eval_args=eval_args, 
            tokenizer=tokenizer, 
            **kwargs
        )
    else:
        dataset = get_dataset(
            train_args, model_args, data_args, eval_args, tokenizer, **kwargs
        )
    kwargs.update(dataset)
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_args=train_args,
        **kwargs,
    )
    trainer.train()

        