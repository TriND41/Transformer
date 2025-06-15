import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils.common import get_eps
from typing import Optional, Union, Tuple

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout_p: float = 0.0, bias: bool = True) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.sqrt_dim = math.sqrt(self.head_dim)

        # QKV Projection Layers
        self.q_proj = nn.Linear(in_features=d_model, out_features=d_model, bias=bias)
        self.k_proj = nn.Linear(in_features=d_model, out_features=d_model, bias=bias)
        self.v_proj = nn.Linear(in_features=d_model, out_features=d_model, bias=bias)

        # Attention Dropout Layer
        self.dropout = nn.Dropout(p=dropout_p)

        # Output Projection Layer
        self.out_proj = nn.Linear(in_features=d_model, out_features=d_model, bias=bias)

    def scaled_dot_product_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, get_weights: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # Q x K^T
        attn = torch.matmul(q, k.transpose(2, 3))

        # Scale
        attn /= self.sqrt_dim

        # (Optional) Mask
        if attn_mask is not None:
            attn.masked_fill_(attn_mask, get_eps(attn))

        # Softmax
        weights = F.softmax(attn, dim=3)
        weights = self.dropout(weights)

        # Compute Attention Weights
        attn = torch.matmul(weights, v)
        
        if not get_weights:
            return attn
        else:
            return attn, weights
        
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, get_weights: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
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
        weights = None
        if not get_weights:
            attn = self.scaled_dot_product_attention(q, k, v, attn_mask, get_weights=get_weights)
        else:
            attn, weights = self.scaled_dot_product_attention(q, k, v, attn_mask, get_weights=get_weights)

        # Attention Projection
        attn = attn.transpose(1, 2).contiguous().view([batch_size, query_length, self.d_model])
        attn = self.out_proj(attn)

        attn, weights