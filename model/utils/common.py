import torch

def get_eps(
    x: torch.Tensor,
    eps16: float = torch.finfo(torch.float16).min,
    eps32: float = torch.finfo(torch.float32).min,
    eps64: float = torch.finfo(torch.float64).min
) -> float:
    if x.dtype == torch.float16:
        return eps16
    elif x.dtype == torch.float32:
        return eps32
    elif x.dtype == torch.float64:
        return eps64
    else:
        return -float('inf')
    
def extend_look_ahead_mask(padding_mask: torch.Tensor) -> torch.Tensor:
    length = padding_mask.size(1)
    trig_matrix = torch.tril(torch.ones([length, length], device=padding_mask.device), diagonal=0).unsqueeze(0)
    return torch.logical_and(padding_mask.unsqueeze(1).repeat([1, length, 1]), trig_matrix)