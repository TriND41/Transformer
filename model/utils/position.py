import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.register_buffer(
            name='angles',
            tensor=1.0 / torch.pow(10000, torch.arange(0, embedding_dim, step=2) / embedding_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.size(1)
        positions = torch.arange(length, dtype=x.dtype, device=x.device).unsqueeze(1) # [length, 1]
        angles = torch.matmul(positions, self.angles.unsqueeze(0)) # [length, d_model/2]

        pe = torch.zeros([length, self.embedding_dim], dtype=x.dtype, device=x.device)
        pe[:, 0::2] = torch.sin(angles)
        pe[:, 1::2] = torch.cos(angles)

        return pe.unsqueeze(0)