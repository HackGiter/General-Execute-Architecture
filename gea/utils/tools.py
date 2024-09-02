import os
import re
import shutil
from typing import List
from pathlib import Path

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
        logger.info(f"Deleting older checkpoint [{ckpt.split('/')[-1]}] due to save_total_limit:{save_total_limit}", extra={"prefix":"\n\r"})
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