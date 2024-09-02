import sys
from typing import Optional, Tuple, Dict, Any
from .data_args import DataArguments
from .eval_args import EvalArguments
from .train_args import TrainArguments
from .model_args import ModelArguments

from transformers import HfArgumentParser

def parse_args(args: Optional[Dict[str, Any]] = None) -> Tuple[ModelArguments, DataArguments, TrainArguments, EvalArguments, str]:
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainArguments, EvalArguments))
    
    if args is not None:
        model_args, data_args, train_args, eval_args = parser.parse_dict(args)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        model_args, data_args, train_args, eval_args = parser.parse_yaml_file(sys.argv[1])
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, train_args, eval_args = parser.parse_json_file(sys.argv[1])
    model_args, data_args, train_args, eval_args, unknown_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    if unknown_args:
        print(parser.format_help())
        print("Got unknown args: {}".format(unknown_args))
        raise ValueError("Some specified arguments are not used by the HfArgumentParser: {}".format(unknown_args))

    return (model_args, data_args, train_args, eval_args, unknown_args)