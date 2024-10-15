import os
import re
import shutil
from typing import Union, Tuple, List, Any
from pathlib import Path

import torch.nn as nn
from transformers import AutoModel

from .constant import ALL_LAYERNORM_LAYERS
from .logging import get_logger

logger = get_logger(__name__)

def sorted_checkpoints(output_dir:str = None, checkpoint_prefix="checkpoint", regex_pattern:str=r".*checkpoint-\d+-([0-9]+)") -> List[str]:
    ckpt_sorted = []

    global_checkpoints = [str(x) for x in Path(output_dir).glob(f"{checkpoint_prefix}-*") if os.path.isdir(x)]
    for path in global_checkpoints:
        regex_match = re.match(regex_pattern, path)
        if regex_match is not None and regex_match.groups() is not None:
            ckpt_sorted.append((int(regex_match.group(1)), path))
    
    ckpt_sorted = [ckpt[1] for ckpt in sorted(ckpt_sorted)]
    return ckpt_sorted

def rotate_checkpoints(save_total_limit:int = None, output_dir:str = None, checkpoint_prefix="checkpoint", regex_pattern:str=r".*checkpoint-\d+-([0-9]+)") -> None:
    if save_total_limit is None or save_total_limit <= 0:
        return
    
    ckpt_sorted = sorted_checkpoints(output_dir, checkpoint_prefix, regex_pattern)
    if len(ckpt_sorted) < save_total_limit:
        return
    
    ckpts_removed = ckpt_sorted[:max(0, len(ckpt_sorted) - save_total_limit)]
    for ckpt in ckpts_removed:
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
            for n in get_parameter_names(child, forbidden_layer_types, forbidden_layer_names, forbidden_module)
            if not (
                isinstance(child, tuple(forbidden_layer_types))
                or (child in tuple(forbidden_module) if forbidden_module is not None else False)
                or (name in forbidden_layer_names if forbidden_layer_names is not None else False)
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

    if isinstance(model, nn.ModuleList):
        prev_mod_str, prev_mod_key, prev_cnt_mod = None, None, 1
        for key, module in model._modules.items():
            mod_str = get_model_details(module, details)
            mod_str = _addindent(mod_str, 2)
            if mod_str != prev_mod_str and prev_mod_str is not None:
                if prev_cnt_mod == 1:
                    child_lines.append('(' + prev_mod_key + '): ' + prev_mod_str)
                else:
                    child_lines.append(f'(0-{prev_cnt_mod-1}) {prev_cnt_mod} x ' + prev_mod_str)
                prev_cnt_mod = 1
            else:
                prev_cnt_mod += 1
            prev_mod_str, prev_mod_key = mod_str, key
        if prev_mod_str is not None:
            if prev_cnt_mod == 1:
                child_lines.append('(' + prev_mod_key + '): ' + prev_mod_str)
            else:
                child_lines.append(f'(0-{prev_cnt_mod-2:}) {prev_cnt_mod - 1} x ' + prev_mod_str)
    else:
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

def count_parameters(model: nn.Module) -> Tuple[int, int]:
    r"""
    Returns the number of trainable parameters and number of all parameters in the model.
    """
    trainable_params, all_param = 0, 0
    for param in model.parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        # Due to the design of 4bit linear layers from bitsandbytes, multiply the number of parameters by itemsize
        if param.__class__.__name__ == "Params4bit":
            if hasattr(param, "quant_storage") and hasattr(param.quant_storage, "itemsize"):
                num_bytes = param.quant_storage.itemsize
            elif hasattr(param, "element_size"):  # for older pytorch version
                num_bytes = param.element_size()
            else:
                num_bytes = 1

            num_params = num_params * 2 * num_bytes

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return trainable_params, all_param

def get_decay_parameter_names(model, forbidden_layer_types:List[Any], forbidden_layer_names:List[str]=None, forbidden_module:List[Any]=None) -> List[str]:
    """
    Get all parameter names that weight decay will be applied to

    Note that some models implement their own layernorm instead of calling nn.LayerNorm, weight decay could still
    apply to those modules since this function only filter out instance of nn.LayerNorm
    """
    decay_parameters = get_parameter_names(model, forbidden_layer_types, forbidden_layer_names, forbidden_module)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    return decay_parameters