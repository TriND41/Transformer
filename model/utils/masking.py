import torch

def generate_padding_mask(lengths: torch.Tensor) -> torch.Tensor:
    positions = torch.arange(lengths.max().item(), dtype=lengths.dtype, device=lengths.device)
    return lengths.unsqueeze(1) > positions.unsqueeze(0)
    
def extend_look_ahead_mask(padding_mask: torch.Tensor) -> torch.Tensor:
    length = padding_mask.size(1)
    trig_matrix = torch.tril(torch.ones([length, length], device=padding_mask.device), diagonal=0).unsqueeze(0)
    return torch.logical_and(padding_mask.unsqueeze(1).repeat([1, length, 1]), trig_matrix)