import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, d_model: int, n_factors: int = 4, bias: bool = True) -> None:
        super().__init__()
        d_ff = d_model * n_factors

        self.hidden_layer = nn.Linear(in_features=d_model, out_features=d_ff, bias=bias)
        self.activation = nn.ReLU()
        self.out_layer = nn.Linear(in_features=d_ff, out_features=d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.hidden_layer(x)
        x = self.activation(x)
        x = self.out_layer(x)
        return x