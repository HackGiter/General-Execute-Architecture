import os
import sys
import logging
from typing import Any
from collections.abc import MutableMapping

import torch.distributed as dist
from logging import LoggerAdapter
from accelerate.logging import MultiProcessAdapter

# 定义颜色常量
class LogColors:
    ERROR = "\033[31m"   # 红色
    INFO = "\033[32m"    # 绿色
    WARN = "\033[33m"    # 黄色
    DEBUG = "\033[34m"   # 蓝色
    RESET = "\033[0m"    # 重置颜色

# 自定义 Formatter 类，用于根据日志级别设置颜色
class LevelFormatter(logging.Formatter):
    def format(self, record):
        # 根据日志级别设置不同的颜色
        record.prefix = getattr(record, 'prefix', '')
        if record.levelno == logging.INFO:
            record.levelname = f"{LogColors.INFO}{record.levelname}{LogColors.RESET}"
        elif record.levelno == logging.WARNING:
            record.levelname = f"{LogColors.WARN}{record.levelname}{LogColors.RESET}"
        elif record.levelno == logging.ERROR:
            record.levelname = f"{LogColors.ERROR}{record.levelname}{LogColors.RESET}"
        elif record.levelno == logging.DEBUG:
            record.levelname = f"{LogColors.DEBUG}{record.levelname}{LogColors.RESET}"
        return super().format(record)

class MultiProcessAdapterWithExtra(MultiProcessAdapter):
    def process(self, msg: Any, kwargs: MutableMapping[str, Any]) -> tuple[Any, MutableMapping[str, Any]]:
        kwargs['extra'] = kwargs.pop('extra', {})
        kwargs['extra'].update(self.extra)
        return msg, kwargs
    
class LoggerAdapterWithExtra(LoggerAdapter):
    def process(self, msg: Any, kwargs: MutableMapping[str, Any]) -> tuple[Any, MutableMapping[str, Any]]:
        kwargs['extra'] = kwargs.pop('extra', {})
        kwargs['extra'].update(self.extra)
        return msg, kwargs

def get_logger(name: str = None, log_level:str = None) -> logging.LoggerAdapter:
    if name is None:
        name = __name__.split(".")[0]
    logger = logging.getLogger(name)
    # 创建控制台处理器
    if log_level is None:
        log_level = os.environ.get("ACCELERATE_LOG_LEVEL", None)
    if log_level is not None:
        logger.setLevel(log_level.upper())
        logger.root.setLevel(log_level.upper())
    else:
        logger.setLevel(logging.DEBUG)
        logger.root.setLevel(logging.DEBUG)
    if not logger.hasHandlers():
        formatter = LevelFormatter(
            fmt="%(prefix)s[%(levelname)s|%(name)s:%(lineno)s] %(asctime)s >> %(message)s", 
            datefmt="%m/%d/%Y %H:%M:%S"
        )
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        handler.setLevel(logger.level)
        logger.addHandler(handler)
    if dist.is_initialized():
        return MultiProcessAdapterWithExtra(logger, {})
    else:
        return LoggerAdapterWithExtra(logger, {})
