import torch
import torch.nn as nn
from typing import Optional

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim: int, max_positions: Optional[int] = None) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        if max_positions is None:
            pe = torch.empty(0)
        else:
            pe = self.__encode(max_positions)
        self.register_buffer('pe', pe)

    def __encode(self, length: int, device: str = 'cpu') -> torch.Tensor:
        div_term = torch.arange(0, self.embedding_dim, 2, device=device).unsqueeze(0)
        positions = torch.arange(length, dtype=div_term.dtype, device=device).unsqueeze(1)
        angles = torch.matmul(positions, div_term)

        pe = torch.zeros([length, self.embedding_dim], dtype=div_term.dtype, device=device)
        pe[:, 0::2] = torch.sin(angles)
        pe[:, 1::2] = torch.cos(angles)

        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.size(1)
        if self.pe.numel() == 0 or length > self.pe.size(0):
            pe = self.__encode(length, device=x.device)
            self.pe = pe
        return self.pe[:length].unsqueeze(dim=0)