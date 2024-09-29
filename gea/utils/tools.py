import os
import re
import shutil
from typing import Union, Tuple, List, Any
from pathlib import Path

import torch.nn as nn
from transformers import AutoModel

from .logging import get_logger

logger = get_logger(__name__)

def sorted_checkpoints(output_dir:str = None, checkpoint_prefix="checkpoint", regex_pattern:str=".*checkpoint-\d+-([0-9]+)") -> List[str]:
    ckpt_sorted = []

    global_checkpoints = [str(x) for x in Path(output_dir).glob(f"{checkpoint_prefix}-*") if os.path.isdir(x)]
    for path in global_checkpoints:
        regex_match = re.match(regex_pattern, path)
        if regex_match is not None and regex_match.groups() is not None:
            ckpt_sorted.append((int(regex_match.group(1)), path))
    
    ckpt_sorted = [ckpt[1] for ckpt in sorted(ckpt_sorted)]
    return ckpt_sorted

def rotate_checkpoints(save_total_limit:int = None, output_dir:str = None, checkpoint_prefix="checkpoint", regex_pattern:str=".*checkpoint-\d+-([0-9]+)") -> None:
    if save_total_limit is None or save_total_limit <= 0:
        return
    
    ckpt_sorted = sorted_checkpoints(output_dir, checkpoint_prefix, regex_pattern)
    if len(ckpt_sorted) < save_total_limit:
        return
    
    ckpts_removed = ckpt_sorted[:max(0, len(ckpt_sorted) - save_total_limit)]
    for ckpt in ckpts_removed:
        # logger.info(f"Deleting older checkpoint [{ckpt.split('/')[-1]}] due to save_total_limit:{save_total_limit}", extra={"prefix":"\n\r"})
        logger.info(f"Deleting older checkpoint [{ckpt.split('/')[-1]}] due to save_total_limit:{save_total_limit}")
        shutil.rmtree(ckpt, ignore_errors=True)

def handle_unknown_kwargs(unknown_kwargs:List[str]) -> str:
    unknown_kwargs_dict = {}
    for i, item in enumerate(unknown_kwargs):
        if item.startswith("--"):
            if i == len(unknown_kwargs) - 1 or unknown_kwargs[i + 1].startswith("--"):
                unknown_kwargs_dict[item.replace("--", "").strip()] = True
            else:
                unknown_kwargs_dict[item] = unknown_kwargs[i + 1]
    return str(unknown_kwargs_dict)

def get_parameter_names(model:Union[AutoModel, nn.Module], forbidden_layer_types:List[Any], forbidden_layer_names:List[str]=None, forbidden_module:List[Any]=None):
    """
    Returns the names of the model parameters that are not inside a forbidden layer or forbidden module.
    Can be used to get a subset of parameter names for decay masks, or to exclude parameters from an optimiser
    (e.g. if the module is frozen).
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types, forbidden_module)
            if not (
                isinstance(child, tuple(forbidden_layer_types))
                or (child in tuple(forbidden_module) if forbidden_module is not None else False)
                or (name not in forbidden_layer_names if forbidden_layer_names is not None else False)
            )
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result

def get_model_details(model:Union[AutoModel, nn.Module], details:bool=False) -> str:
    def _addindent(s_:str, numSpaces:int) -> str:
        s = s_.split('\n')
        # don't do anything for single-line stuff
        if len(s) == 1:
            return s_
        first = s.pop(0)
        s = [(numSpaces * ' ') + line for line in s]
        s = '\n'.join(s)
        s = first + '\n' + s
        return s
    # We treat the extra repr like the sub-module, one item per line
    extra_lines = []
    extra_repr = model.extra_repr()
    # empty string will be split into list ['']
    if extra_repr:
        extra_lines = extra_repr.split('\n')
    child_lines = []
    for key, module in model._modules.items():
        mod_str = get_model_details(module, details)
        mod_str = _addindent(mod_str, 2)
        child_lines.append('(' + key + '): ' + mod_str)
    lines = extra_lines + child_lines

    main_str = model._get_name() + '('
    if lines:
        # simple one-liner info, which most builtin Modules will use
        if len(extra_lines) == 1 and not child_lines:
            main_str += extra_lines[0]
        else:
            main_str += '\n  ' + '\n  '.join(lines) + '\n'

    main_str += ')'
    if (len(extra_lines) == 1 or len(lines) == 0) and details:
        param_infos = " ( "
        for name, param in model.named_parameters():
            param_infos += f"{name}:{param.requires_grad} {param.dtype} {param.device}"
        param_infos += ")"
        main_str += param_infos if len(param_infos) != 4 else ""
    return main_str