import gc
import threading

import torch

from transformers.utils import (
    is_psutil_available,
    is_torch_cuda_available,
    is_torch_npu_available
)

from .callback import StateCallback

class MemoryTracker(StateCallback):
    """
    A helper class that tracks cpu and gpu memory.

    This class will silently skip unless `psutil` is available. Install with `pip install psutil`.

    When a stage completes, it can pass metrics dict to update with the memory metrics gathered during this stage.

    Example :

    ```python
    self._memory_tracker = TrainerMemoryTracker(self.args.skip_memory_metrics)
    self._memory_tracker.start()
    # code ...
    metrics = {"train_runtime": 10.5}
    self._memory_tracker.stop_and_update_metrics(metrics)
    ```

    At the moment GPU tracking is only for `pytorch`, but can be extended to support `tensorflow`.

    To understand this class' intricacies please read the documentation of [`~Trainer.log_metrics`].
    """

    def __init__(self, skip_memory_metrics=False):
        self.skip_memory_metrics = skip_memory_metrics

        if not is_psutil_available():
            # soft dependency on psutil
            self.skip_memory_metrics = True

        if self.skip_memory_metrics:
            return

        import psutil  # noqa

        if is_torch_cuda_available():
            import torch

            self.torch = torch
            self.gpu = {}
            self.torch.cuda.reset_peak_memory_stats()
            self.torch.cuda.empty_cache()
        elif is_torch_npu_available():
            import torch

            self.torch = torch
            self.gpu = {}
            self.torch.npu.reset_peak_memory_stats()
            self.torch.npu.empty_cache()
        else:
            self.torch = None

        self.process = psutil.Process()

        self.stages = []
        self.stage_memories = {}
        self.cpu = {}
        self.init_reported = False

    def cpu_mem_used(self):
        """get resident set size memory for the current process"""
        return self.process.memory_info().rss

    def peak_monitor_func(self):
        self.cpu_mem_used_peak = -1

        while True:
            self.cpu_mem_used_peak = max(self.cpu_mem_used(), self.cpu_mem_used_peak)

            # can't sleep or will not catch the peak right (this comment is here on purpose)
            # time.sleep(0.001) # 1msec

            if not self.peak_monitoring:
                break

    def start(self, stage):
        """start tracking for the caller's stage"""
        if self.skip_memory_metrics:
            return

        # deal with nested calls of eval during train - simply ignore those
        if stage not in self.stages:
            self.stages.append(stage)
            self.stage_memories[stage] = {}

        gc.collect()

        # gpu
        if self.torch is not None:
            if torch.cuda.is_available():
                self.stage_memories[stage]["gpu_mem_used_at_start"] = self.torch.cuda.memory_allocated()
            elif is_torch_npu_available():
                self.stage_memories[stage]["gpu_mem_used_at_start"] = self.torch.npu.memory_allocated()

        # cpu
        self.stage_memories[stage]["cpu_mem_used_at_start"] = self.cpu_mem_used()

        self.peak_monitoring = True
        peak_monitor_thread = threading.Thread(target=self.peak_monitor_func)
        peak_monitor_thread.daemon = True
        peak_monitor_thread.start()

    def stop(self, stage):
        """stop tracking for the passed stage"""

        if stage not in self.stages:
            return
        self.stages.remove(stage)

        # this sends a signal to peak_monitor_func to complete its loop
        self.peak_monitoring = False

        # first ensure all objects get collected and their memory is freed
        gc.collect()

        if self.torch is not None:
            if torch.cuda.is_available():
                self.torch.cuda.empty_cache()
            elif is_torch_npu_available():
                self.torch.npu.empty_cache()

        # concepts:
        # - alloc_delta:  the difference of allocated memory between the end and the start
        # - peaked_delta: the difference between the peak memory and the current memory
        # in order to know how much memory the measured code consumed one needs to sum these two

        # gpu
        if self.torch is not None:
            if torch.cuda.is_available():
                self.stage_memories[stage]["gpu_mem_used_now"] = self.torch.cuda.memory_allocated()
                self.stage_memories[stage]["gpu_mem_used_peak"] = self.torch.cuda.max_memory_allocated()
            elif is_torch_npu_available():
                self.stage_memories[stage]["gpu_mem_used_now"] = self.torch.npu.memory_allocated()
                self.stage_memories[stage]["gpu_mem_used_peak"] = self.torch.npu.max_memory_allocated()

            self.gpu[stage] = {
                "begin": self.stage_memories[stage]["gpu_mem_used_at_start"],
                "end": self.stage_memories[stage]["gpu_mem_used_now"],
                "alloc": (self.stage_memories[stage]["gpu_mem_used_now"] - self.stage_memories[stage]["gpu_mem_used_at_start"]),
            }
            if self.gpu_mem_used_peak is not None:
                self.gpu[stage]["peaked"] = max(0, self.stage_memories[stage]["gpu_mem_used_peak"] - self.stage_memories[stage]["gpu_mem_used_now"])
            else:
                self.gpu[stage]["peaked"] = "Not available"

        # cpu
        self.stage_memories[stage]["cpu_mem_used_now"] = self.cpu_mem_used()
        self.cpu[stage] = {
            "begin": self.stage_memories[stage]["cpu_mem_used_at_start"],
            "end": self.stage_memories[stage]["cpu_mem_used_now"],
            "alloc": (self.stage_memories[stage]["cpu_mem_used_now"] - self.stage_memories[stage]["cpu_mem_used_at_start"]),
            "peaked": max(0, self.cpu_mem_used_peak - self.stage_memories[stage]["cpu_mem_used_now"]),
        }

    def update_metrics(self, stage, metrics):
        """updates the metrics"""
        if self.skip_memory_metrics:
            return

        # since we don't have a way to return init metrics, we push them into the first of train/val/predict
        stages = [stage]
        if not self.init_reported:
            stages.insert(0, "init")
            self.init_reported = True

        for stage in stages:
            for t in ["alloc", "peaked"]:
                if stage in self.cpu and t in self.cpu[stage]:
                    metrics[f"{stage}_mem_cpu_{t}_delta"] = self.cpu[stage][t]
                if self.torch is not None and stage in self.gpu and t in self.gpu[stage]:
                    metrics[f"{stage}_mem_gpu_{t}_delta"] = self.gpu[stage][t]
            # if we need additional debug info, enable the following
            # for t in ["begin", "end"]:
            #     if stage in self.cpu and t in self.cpu[stage]:
            #         metrics[f"{stage}_mem_cpu_{t}"] = self.cpu[stage][t]
            #     if self.torch is not None and stage in self.gpu and t in self.gpu[stage]:
            #         metrics[f"{stage}_mem_gpu_{t}"] = self.gpu[stage][t]

        # since memory can be allocated before init, and it might be difficult to track overall
        # memory usage, in particular for GPU, let's report memory usage at the point init was called
        if stages[0] == "init":
            metrics["before_init_mem_cpu"] = self.cpu["init"]["begin"]
            if self.torch is not None:
                metrics["before_init_mem_gpu"] = self.gpu["init"]["begin"]
            # if we also wanted to report any additional memory allocations in between init and
            # whatever the next stage was we could also report this:
            # if self.cpu["init"]["end"] != self.cpu[stage]["begin"]:
            #     metrics[f"after_init_mem_cpu_delta"] = self.cpu[stage]["begin"] - self.cpu["init"]["end"]
            # if self.torch is not None and self.gpu["init"]["end"] != self.gpu[stage]["begin"]:
            #     metrics[f"after_init_mem_gpu_delta"] = self.gpu[stage]["begin"] - self.gpu["init"]["end"]

    def stop_and_update_metrics(self, stage, metrics=None):
        """combine stop and metrics update in one call for simpler code"""
        if self.skip_memory_metrics:
            return

        self.stop(stage)

        # init doesn't have metrics to update so we just save that data for later stages to retrieve
        if metrics is not None:
            self.update_metrics(stage, metrics)
