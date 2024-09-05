from enum import Enum
from typing import Callable, Union, Dict, List, Any

import torch
import numpy as np

from .distance import cosine_distance, absolute_distance

class Method(str, Enum):
    MEAN = "mean"
    MAX = "max"
    MIN = "min"
    SUM = "sum"

class DistanceMetric(str, Enum):
    COSINE = "cosine"
    ABSOLUTE = "absolute"
    EUCLIDEAN = "euclidean"
    MINKOWSKI = "MINKOWSKI"

DISTANCE_FUNCTIONS = {
    DistanceMetric.COSINE: cosine_distance,
    DistanceMetric.ABSOLUTE: absolute_distance,
    DistanceMetric.EUCLIDEAN: None,
    DistanceMetric.MINKOWSKI: None,
}

class MetricBase:
    _merge_times: int = 1
    def __init__(self) -> None:
        pass

    def start(self, **kwargs) -> None:
        pass

    def end(self, **kwargs) -> None:
        pass

    def update(self, **kwargs) -> None:
        pass

    def compute(self, **kwargs) -> None:
        pass

    def summary(self, **kwargs) -> None:
        pass

    def merge(self, **kwargs) -> None:
        pass

class Timer(MetricBase):
    timing: float = None
    _starter: torch.cuda.Event = None
    _ender: torch.cuda.Event = None

    def __init__(self) -> None:
        self._starter = torch.cuda.Event(enable_timing=True)
        self._ender = torch.cuda.Event(enable_timing=True)
        super().__init__()

    def start(self) -> None:
        """
        Start the time record
        """
        torch.cuda.synchronize()
        self._starter.record()

    def end(self) -> None:
        """
        End the time record
        """
        self._ender.record()
        torch.cuda.synchronize()
        self.timing = self._starter.elapsed_time(self._ender)

    def summary(self) -> Dict[str, Any]:
        return { "timing": self.timing }
    
    def merge(self, eval:'Timer', method: Union[Method, str] = Method.MEAN) -> None:
        _merge_times = (self._merge_times + eval._merge_times)
        if method == Method.MEAN:
            self.timing = (self.timing * self._merge_times + eval.timing * self._merge_times) / _merge_times
        elif method == Method.MAX:
            self.timing = max(self.timing, eval.timing)
        elif method == Method.MIN:
            self.timing = min(self.timing, eval.timing)
        else:
            self.timing = self.timing + eval.timing
        self._merge_times = _merge_times

class  AssistantMetric(Timer):
    k: int = 1
    step: int = 1
    ratio: float = 0
    tokens: int = None
    accepts: List[int] = None
    acc_num: List[int] = None
    accuracy: List[float] = None
    
    def __init__(self, k: int, tokens: int) -> None:
        self.k = k
        self.tokens = tokens
        self.accepts = [0] * (k + 1)
        super().__init__()

    def update(self, matches:Union[int, torch.Tensor]) -> None:
        """
        Update some metric factors used for compute
        """
        self.ratio = self.ratio * (self.step - 1) / self.step + (matches / self.k) / self.step
        self.accepts[matches] += 1
        self.step += 1

    def compute(self, tokens:int) -> None:
        """
        Calculate the metric summary of all samples
        """
        self.acc_num = [
            self.accepts[i] if i == 0 else sum(self.accepts[i:])
            for i in range(self.k + 1)
        ]
        self.accuracy = [
            self.acc_num[i + 1] / (sum(self.accepts if i == 0 else self.accepts[i:]) + 1e-12) 
            for i in range(self.k)
        ]
        self.tokens = tokens - self.tokens

    def summary(self) -> Dict[str, Any]:
        return {
            "timing": self.timing,
            "ratio": self.ratio,
            "acc_num": self.acc_num,
            "accuracy": self.accuracy,
            "tok_per_sec": self.timing / self.tokens,
        }
    
    def merge(self, eval:'AssistantMetric', method: Union[Method, str] = Method.MEAN) -> None:
        _merge_times = (self._merge_times + eval._merge_times)
        if method == Method.MEAN:
            self.ratio = (self.ratio * self._merge_times + eval.ratio * eval._merge_times) / _merge_times
            self.acc_num = [(it1 * self._merge_times + it2 * eval._merge_times) / _merge_times for it1, it2 in zip(self.acc_num, eval.acc_num)]
            self.accuracy = [(it1 * self._merge_times + it2 * eval._merge_times) / _merge_times for it1, it2 in zip(self.accuracy, eval.accuracy)]
            self.tokens = (self.tokens * self._merge_times + eval.tokens * eval._merge_times) / _merge_times
            self.timing = (self.timing * self._merge_times + eval.timing * self._merge_times) / _merge_times
        elif method == Method.MAX:
            self.ratio = max(self.ratio, eval.ratio)
            self.acc_num = max(self.acc_num, eval.acc_num)
            self.accuracy = max(self.accuracy, eval.accuracy)
            self.tokens = max(self.tokens, eval.tokens)
            self.timing = max(self.timing, eval.timing)
        elif method == Method.MIN:
            self.ratio = min(self.ratio, eval.ratio)
            self.acc_num = min(self.acc_num, eval.acc_num)
            self.accuracy = min(self.accuracy, eval.accuracy)
            self.tokens = min(self.tokens, eval.tokens)
            self.timing = min(self.timing, eval.timing)
        else:
            self.ratio = self.ratio + eval.ratio
            self.acc_num = self.acc_num + eval.acc_num
            self.accuracy = self.accuracy + eval.accuracy
            self.tokens = self.tokens + eval.tokens
            self.timing = self.timing + eval.timing
        self._merge_times = _merge_times

class StateMetric(MetricBase):
    def __init__(
            self, 
            metric_fns:Union[Callable, DistanceMetric, str, List[Callable], List[DistanceMetric], List[str]] = None) -> None:
        super().__init__()
        self.metric_fns = []
        if not isinstance(metric_fns, List):
            metric_fns = [metric_fns]
        for item in metric_fns:
            if isinstance(item, Callable):
                self.metric_fns.append(item)
            else:
                self.metric_fns.append(DISTANCE_FUNCTIONS[item])
        self.results = [[] for _ in range(len(self.metric_fns))]

    def update(self, x:torch.Tensor, y:torch.Tensor = None, **kwargs) -> None:
        for i, metric_fn in enumerate(self.metric_fns):
            result = metric_fn(x, y, **kwargs).mean(dim=0)
            self.results[i].append(result.cpu().numpy() if isinstance(result, torch.Tensor) else result)

    def compute(self) -> None:
        self.results = [np.array(item).mean(axis=0) if len(item) != 0 else [] for item in self.results]

    def summary(self) -> Dict[str, Any]:
        return { "metrics" : self.results }          

    def merge(self, eval:'StateMetric', method: Union[Method, str] = Method.MEAN) -> None:
        _merge_times = (self._merge_times + eval._merge_times)
        if len(self.results[0]) != 0 and len(eval.results[0]) != 0:
            if method == Method.MEAN:
                self.results = [(it1 * self._merge_times + it2 * eval._merge_times) / _merge_times for it1, it2 in zip(self.results, eval.results)]
            elif method == Method.MAX:
                self.results = [np.maximum(it1, it2) for it1, it2 in zip(self.results, eval.results)]
            elif method == Method.MIN:
                self.results = [np.minimum(it1, it2) for it1, it2 in zip(self.results, eval.results)]
            else:
                self.results = [it1 + it2 for it1, it2 in zip(self.results, eval.results)]
            self._merge_times = _merge_times
        elif len(eval.results[0]) != 0:
            self._merge_times = eval._merge_times
            self.results = eval.results

class TokenStatisticMetric(MetricBase):
    def __init__(
            self, 
            idxs:int = None,
            vocab_size:int = None, ) -> None:
        super().__init__()
        self.idxs = idxs
        self.vocab_size = vocab_size
        self.results = np.zeros([self.idxs + 1, self.vocab_size], dtype=np.float32)

    def update(self, id:int = None, label:int = None, **kwargs) -> None:
        self.results[-1, label] += 1
        self.results[id, label] += 1

    def summary(self) -> Dict[str, Any]:
        return { "metrics" : self.results }          

    def merge(self, eval:'TokenStatisticMetric', method: Union[Method, str] = Method.MEAN) -> None:
        _merge_times = (self._merge_times + eval._merge_times)
        if method == Method.MEAN:
            self.results = (self.results * self._merge_times + eval.results * eval._merge_times) / _merge_times
        elif method == Method.MAX:
            self.results = np.maximum(self.results, eval.results)
        elif method == Method.MIN:
            self.results = np.minimum(self.results, eval.results)
        else:
            self.results = self.results + eval.results
        self._merge_times = _merge_times
