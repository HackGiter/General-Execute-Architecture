from tqdm import tqdm
from typing import Iterable, Union, Tuple, Dict, List, Any

import torch
import numpy as np

from ..metric.metrics import Method, MetricBase
from ..utils.logging import get_logger

logger = get_logger(__name__)

def process_metrics(
        summaries: Union[List[Any], Dict[str, Any], Any] = None, 
        metrics: Union[List[Any], Dict[str, Any], Any] = None) -> Any:
    if summaries is None:
        if not isinstance(metrics, Dict) and not isinstance(metrics, List):
            return [metrics.cpu()] if isinstance(metrics, torch.Tensor) else [metrics]

    if isinstance(metrics, Dict):
        summaries = {} if summaries is None else summaries
        for key, value in metrics.items():
            summaries[key] = process_metrics(summaries.get(key, None), value)
        return summaries
    elif isinstance(metrics, List):
        if summaries is None:
            summaries = []
            for item in metrics:
                summaries.append(process_metrics(None, item))
        else:
            for i, item in enumerate(metrics):
                summaries[i] = process_metrics(summaries[i], item)
        return summaries
    elif isinstance(metrics, torch.Tensor):
        return summaries + [metrics.cpu()]
    else:
        return (summaries if summaries is not None else []) + [np.array(metrics)]
    
def process_method(metrics: np.ndarray, method:Method) -> np.ndarray:
    if method == Method.MEAN:
        return metrics.mean(axis=0)
    elif method == Method.MAX:
        return metrics.max(axis=0)
    elif method == Method.MIN:
        return metrics.min(axis=0)
    else:
        return metrics.sum(axis=0)

def summary_metrics(metrics: Union[List[Any], Dict[str, Any], Any], method:Method) -> Any:
    if isinstance(metrics, Dict):
        return { k:summary_metrics(v, method) for k, v in metrics.items() }
    elif isinstance(metrics, List):
        if len(metrics) > 0:
            if not isinstance(metrics[0], List) and not isinstance(metrics[0], Dict):
                return process_method(np.array(metrics), method)
            else:
                return [summary_metrics(item, method) for item in metrics]
    return metrics

class Experiments:
    records: List[str] = None
    num: int = None
    metrics: Union[List[MetricBase], Tuple[MetricBase]] = None
    
    def __init__(self, num:int, records: Union[List[str], Tuple[str]] = None, **kwargs) -> None:
        self.num = num
        self.summaries = None
        self.records = records if records is not None else []
        for key, value in kwargs.items():
            setattr(self, key, value)
        for key in self.records:
            setattr(self, key, [])
    
    def add(self, eval: Union[MetricBase, Any]) -> bool:
        _intermediate = getattr(self, 'intermediate', None)
        if isinstance(eval, MetricBase):
            if _intermediate is not None:
                eval.merge(_intermediate)
                del self._intermediate
            if self.metrics is None:
                self.metrics = [eval]
            else:
                self.metrics.append(eval)
        else:
            _metrics:MetricBase = getattr(eval, 'metrics', None)
            if _intermediate is not None and _metrics is not None:
                _metrics.merge(_intermediate)
                del self._intermediate
            if self.metrics is None:
                self.metrics = [_metrics]
            else:
                self.metrics.append(_metrics)
            for key in self.records:
                getattr(self, key).append(getattr(eval, key, None))
        if getattr(self, '_progressing', None) is not None:
            self._progressing.update(1)
        if len(self.metrics) == self.num:
            self._progressing.close()
            return True
        else:
            return False
        
    def merge(self, eval: Union[MetricBase, Any], method: Union[Method, str] = Method.MEAN) -> None:
        if isinstance(eval, MetricBase):
            if getattr(self, '_intermediate', None) is None:
                self._intermediate = eval
            else:
                self._intermediate.merge(eval, method)
        else:
            if getattr(self, '_intermediate', None) is None:
                self._intermediate = getattr(eval, 'metrics', None)
            else:
                self._intermediate.merge(getattr(eval, 'metrics', None), method)

    def log(self, only_setup: bool=True, file_name: str="setup.json") -> bool:
        import json
        try:
            with open(file_name, 'r') as f:
                obj = vars(self)
                if only_setup:
                    obj.pop("metrics")
                json.dump(obj, f)
                return True
        except Exception as e:
            logger.error(f"Raise Exception: {e}")
            return False

    # def summary(self) -> Dict[str, List[Any]]:
    #     self.summaries = dict()
    #     for item in self.metrics:
    #         for key, value in item.summary().items():
    #             if self.summaries.get(key, None) is None:
    #                 self.summaries[key] = [value.cpu()] if isinstance(value, torch.Tensor) else [value]
    #             else:
    #                 self.summaries[key].append(value.cpu() if isinstance(value, torch.Tensor) else value)
    #     # logger.debug(self.summaries.keys())
    #     print(self.summaries.keys())
    #     for key, value in self.summaries.items():
    #         if isinstance(value, Iterable):
    #             self.summaries[key] = np.array(value).mean(axis=0)
    #         elif isinstance(value, Mapping):
    #             self.summaries[key]

    def summary(self, method:Method = Method.MEAN) -> Dict[str, List[Any]]:
        method = Method.MEAN if method is None else method
        if self.summaries is None:
            for item in self.metrics:
                self.summaries = process_metrics(self.summaries, item.summary())
            # logger.debug(self.summaries.keys())
            self.summaries = summary_metrics(self.summaries, method)
        return self.summaries

    def print(self):
        if self.summaries is None:
            self.summary()
        logger.info("***  METRIC SUMMARIES  ***")
        for key, value in self.summaries.items():
            logs = f"\n* {key.upper():^12} *"
            if isinstance(value, Dict):
                for k, v in value.items():
                    logs += f"\n** {k:^10} **"
                    if isinstance(v, List):
                        for item in v:
                            logs += f"\n{item:.3f}=" if isinstance(item, float) else f"\n{item}"
                    else:
                        logs += f"\n{v:.3f}" if isinstance(v, float) else f"\n{v}"
            elif isinstance(value, List):
                logs += ": "
                for item in value:
                    logs += f"{item:.3f} " if isinstance(item, float) else f"{item} "
            else:
                logs += f": {value:.3f}" if isinstance(value, float) else f"{value}"
            logger.info(logs)

    def tqdm(self, desc:str, total:int=None):
        total = min(total, self.num)
        self._progressing = tqdm(desc=desc, total=total)

    def keys(self):
        # 返回属性名的列表，以便 ** 解包操作使用
        return self.__dict__.keys()

    def __getitem__(self, key):
        # 允许使用 obj[key] 访问属性
        return getattr(self, key)
    
    def __iter__(self):
        # 使实例可以直接通过 ** 解包为字典
        for key, value in self.__dict__.items():
            yield key, value