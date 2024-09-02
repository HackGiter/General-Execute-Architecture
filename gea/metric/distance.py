from typing import Callable, Union
import torch

def standard_norm(x: torch.Tensor, eps:float=1e-12, **kwargs) -> torch.Tensor:
    dtype = x.dtype
    x = x.float()
    mean = x.mean(dim=-1, keepdim=True)
    variance = (x - mean).pow(2).mean(-1, keepdim=True)
    x = (x - mean) * torch.rsqrt(variance + kwargs.get("eps", eps))
    return x.to(dtype)


def cosine_distance(
        x:torch.Tensor, 
        y:torch.Tensor = None, 
        is_mat:bool=True, 
        is_norm:bool=False, 
        norm_fn:Callable=None, 
        **kwargs) -> Union[torch.Tensor, float]:
    x = x.float()
    y = y if y is None else y.float()
    if is_norm:
        x = norm_fn(x, **kwargs) if norm_fn is not None else standard_norm(x, **kwargs)
        if y is not None:
            y = norm_fn(y, **kwargs) if norm_fn is not None else standard_norm(y, **kwargs)
    x = x * torch.rsqrt(x.pow(2).sum(dim=-1, keepdim=True))
    y = y * torch.rsqrt(y.pow(2).sum(dim=-1, keepdim=True)) if y is not None else x
    return torch.matmul(x, y.transpose(-1, -2)) if is_mat else torch.sum(x * y, dim=-1)

def absolute_distance(
        x:torch.Tensor, 
        y:torch.Tensor, 
        is_norm:bool=False, 
        norm_fn:Callable=None, 
        **kwargs) -> Union[torch.Tensor, float]:
    x = x.float()
    y = y if y is None else y.float()
    if is_norm:
        x = norm_fn(x, **kwargs) if norm_fn is not None else standard_norm(x, **kwargs)
        if y is not None:
            y = norm_fn(y, **kwargs) if norm_fn is not None else standard_norm(y, **kwargs)
    if y is not None:
        return (x - y).abs().mean(dim=-1)
    else:
        return (x.clone().unsqueeze(-2) - x.unsqueeze(-3)).abs().mean(dim=-1)

def euclidean_distance(
        x:torch.Tensor, 
        y:torch.Tensor, 
        is_norm:bool=False, 
        norm_fn:Callable=None, **kwargs) -> Union[torch.Tensor, float]:
    x = x.float()
    y = y.float()
    if is_norm:
        x = norm_fn(x, **kwargs) if norm_fn is not None else standard_norm(x, **kwargs)
        y = norm_fn(y, **kwargs) if norm_fn is not None else standard_norm(y, **kwargs)
    return (x - y).pow(2).sum(dim=-1).sqrt()

def minkowski_distance(
        x:torch.Tensor, 
        y:torch.Tensor, 
        is_norm:bool=False, 
        norm_fn:Callable=None, 
        p: float = None,
        **kwargs) -> Union[torch.Tensor, float]:
    x = x.float()
    y = y.float()
    if is_norm:
        x = norm_fn(x, **kwargs) if norm_fn is not None else standard_norm(x, **kwargs)
        y = norm_fn(y, **kwargs) if norm_fn is not None else standard_norm(y, **kwargs)
    return (x - y).pow(p).sum(dim=-1).pow(1 / p)
