import torch
import torch.nn as nn
import torch.nn.functional as F
from model.utils.attention import MultiHeadAttention
from model.utils.ffn import FeedForward
from typing import Optional, Tuple

class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout_p: float, ffn_n_factors: int = 4, attn_bias: bool = True, ffn_bias: bool = True, eps: float = 1e-5) -> None:
        super().__init__()
        self.dropout_p = dropout_p
        
        # Main Layers
        self.attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout_p=dropout_p, bias=attn_bias)
        self.ffn = FeedForward(d_model=d_model, n_factors=ffn_n_factors, bias=ffn_bias)

        # Normalization Layers
        self.attn_norm = nn.LayerNorm(normalized_shape=d_model, eps=eps)
        self.ffn_norm = nn.LayerNorm(normalized_shape=d_model, eps=eps)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Attention
        attn = self.attn(x, x, x, attn_mask)
        attn = self.attn_norm(
            F.dropout(attn, p=self.dropout_p, training=self.training) + x
        )

        # Feed Forward
        ffn = self.ffn(attn)
        ffn = self.ffn_norm(
            F.dropout(ffn, p=self.dropout_p, training=self.training) + attn
        )

        return ffn
        
class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout_p: float, ffn_n_factors: int = 4, attn_bias: bool = True, ffn_bias: bool = True, eps: float = 1e-5) -> None:
        super().__init__()
        self.dropout_p = dropout_p

        # Main Layers
        self.masked_attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout_p=dropout_p, bias=attn_bias)
        self.cross_attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout_p=dropout_p, bias=attn_bias)
        self.ffn = FeedForward(d_model=d_model, n_factors=ffn_n_factors, bias=ffn_bias)

        # Normalization Layers
        self.masked_attn_norm = nn.LayerNorm(normalized_shape=d_model, eps=eps)
        self.cross_attn_norm = nn.LayerNorm(normalized_shape=d_model, eps=eps)
        self.ffn_norm = nn.LayerNorm(normalized_shape=d_model, eps=eps)

    def get_kv_cache(self, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        k, v = self.cross_attn.get_kv_cache(context, context)
        return k, v
    
    def forward_kv_cache(
        self, 
        x: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        masked_attn = self.masked_attn(x, x, x, attn_mask)
        masked_attn = self.masked_attn_norm(
            F.dropout(masked_attn, p=self.dropout_p, training=self.training) + x
        )

        # Cross Attention
        cross_attn = self.cross_attn.forward_kv_cache(masked_attn, k, v, context_mask)
        cross_attn = self.cross_attn_norm(
            F.dropout(cross_attn, p=self.dropout_p, training=self.training) + attn_mask
        )

        # Feed Forward
        ffn = self.ffn(cross_attn)
        ffn = self.ffn_norm(
            F.dropout(ffn, p=self.dropout_p, training=self.training) + cross_attn
        )

        return ffn

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Masked Attention
        masked_attn = self.masked_attn(x, x, x, attn_mask)
        masked_attn = self.masked_attn_norm(
            F.dropout(masked_attn, p=self.dropout_p, training=self.training) + x
        )

        # Cross Attention
        cross_attn = self.cross_attn(masked_attn, context, context, context_mask)
        cross_attn = self.cross_attn_norm(
            F.dropout(cross_attn, p=self.dropout_p, training=self.training) + attn_mask
        )

        # Feed Forward
        ffn = self.ffn(cross_attn)
        ffn = self.ffn_norm(
            F.dropout(ffn, p=self.dropout_p, training=self.training) + cross_attn
        )

        return ffn