import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

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

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout_p: float = 0.0, bias: bool = True) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale_factor = 1.0 / math.sqrt(self.head_dim)

        # QKV Projection Layers
        self.q_proj = nn.Linear(in_features=d_model, out_features=d_model, bias=bias)
        self.k_proj = nn.Linear(in_features=d_model, out_features=d_model, bias=bias)
        self.v_proj = nn.Linear(in_features=d_model, out_features=d_model, bias=bias)

        # Attention Dropout Layer
        self.dropout = nn.Dropout(p=dropout_p)

        # Output Projection Layer
        self.out_proj = nn.Linear(in_features=d_model, out_features=d_model, bias=bias)

    def scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Q x K^T / sqrt(dk)
        attn_scores = torch.matmul(q, k.transpose(2, 3)) * self.scale_factor

        # (Optional) Mask
        if attn_mask is not None:
            attn_scores.masked_fill_(attn_mask, get_eps(attn_scores))

        # Softmax
        attn_weights = F.softmax(attn_scores, dim=3)
        attn_weights = self.dropout(attn_weights)

        # Compute Attention Weights
        return torch.matmul(attn_weights, v)
        
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, query_length, _ = q.size()
        cross_length = k.size(1)
        
        # QKV Projection
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Split heads
        q = q.view([batch_size, query_length, self.n_heads, self.head_dim]).transpose(1, 2)
        k = k.view([batch_size, cross_length, self.n_heads, self.head_dim]).transpose(1, 2)
        v = v.view([batch_size, cross_length, self.n_heads, self.head_dim]).transpose(1, 2)

        # Compute attention
        attn = self.scaled_dot_product_attention(q, k, v, attn_mask)
        attn = attn.transpose(1, 2).contiguous().view([batch_size, query_length, self.d_model])
        attn = self.out_proj(attn)

        return attn