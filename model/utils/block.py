import torch
import torch.nn as nn
from model.utils.attention import MultiHeadAttention
from model.utils.ffn import FeedForward
from typing import Optional, Union, Tuple

class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout_p: float, ffn_n_factors: int = 4, attn_bias: bool = True, ffn_bias: bool = True, eps: float = 1e-5) -> None:
        super().__init__()
        # Main Layers
        self.attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout_p=dropout_p, bias=attn_bias)
        self.ffn = FeedForward(d_model=d_model, n_factors=ffn_n_factors, bias=ffn_bias)

        # Normalization Layers
        self.attn_norm = nn.LayerNorm(normalized_shape=d_model, eps=eps)
        self.ffn_norm = nn.LayerNorm(normalized_shape=d_model, eps=eps)

        # Dropout for Residual Connection
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Attention
        attn = self.attn(x, x, x, attn_mask)
        attn = self.attn_norm(self.dropout(attn) + x)

        # Feed Forward
        ffn = self.ffn(attn)
        ffn = self.ffn_norm(self.dropout(ffn) + attn)

        return ffn
        
class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout_p: float, ffn_n_factors: int = 4, attn_bias: bool = True, ffn_bias: bool = True, eps: float = 1e-5) -> None:
        super().__init__()
        # Main Layers
        self.masked_attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout_p=dropout_p, bias=attn_bias)
        self.cross_attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout_p=dropout_p, bias=attn_bias)
        self.ffn = FeedForward(d_model=d_model, n_factors=ffn_n_factors, bias=ffn_bias)

        # Normalization Layers
        self.masked_attn_norm = nn.LayerNorm(normalized_shape=d_model, eps=eps)
        self.cross_attn_norm = nn.LayerNorm(normalized_shape=d_model, eps=eps)
        self.ffn_norm = nn.LayerNorm(normalized_shape=d_model, eps=eps)

        # Dropout for Residual Connection
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Masked Attention
        masked_attn = self.masked_attn(x, x, x, attn_mask)
        masked_attn = self.masked_attn_norm(self.dropout(masked_attn) + x)

        # Cross Attention
        cross_attn = self.cross_attn(masked_attn, context, context, context_mask)
        cross_attn = self.cross_attn_norm(self.dropout(cross_attn) + masked_attn)

        # Feed Forward
        ffn = self.ffn(cross_attn)
        ffn = self.ffn_norm(self.dropout(ffn) + cross_attn)

        return ffn