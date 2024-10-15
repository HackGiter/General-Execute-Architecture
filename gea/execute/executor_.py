import ast
from typing import Dict, Any

from ..args.parser import parse_args

from ..model.template import get_model
from ..data.process import get_dataset
from ..train.train import Trainer
from ..utils.tools import handle_unknown_kwargs, get_logger

logger = get_logger(__name__)

def run_train(kwargs: Dict[str, Any] = {}):
    model_args, data_args, train_args, eval_args, unknown_args = parse_args()
    unknown_args:Dict[str, Any] = ast.literal_eval(handle_unknown_kwargs(unknown_args))
    kwargs.update(unknown_args)

    model_kwargs = kwargs.pop("model_kwargs", {})
    if kwargs.get("get_model_fn", None) is not None:
        model, tokenizer = kwargs.pop("get_model_fn")(model_args, **model_kwargs)
    else:
        model, tokenizer = get_model(model_args, **model_kwargs)

    data_kwargs = kwargs.pop("data_kwargs", {})
    if kwargs.get("get_dataset_fn", None) is not None:
        dataset = kwargs.pop("get_dataset_fn")(
            train_args=train_args, 
            model_args=model_args, 
            data_args=data_args, 
            eval_args=eval_args, 
            tokenizer=tokenizer, 
            **data_kwargs
        )
    else:
        dataset = get_dataset(
            train_args, model_args, data_args, eval_args, tokenizer, **data_kwargs
        )
    kwargs.update(dataset)
    logger.info(type(dataset["train_dataset"][0]['input_ids']))
    logger.info(dataset["train_dataset"][0])

if __name__ == "__main__":
    run_train()