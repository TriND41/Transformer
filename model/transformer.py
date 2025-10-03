import torch
import torch.nn as nn
from model.modules.encoder import Encoder
from model.modules.decoder import Decoder
from typing import Optional, Tuple, List

class Transformer(nn.Module):
    def __init__(
        self,
        n_encoder_tokens: int,
        n_decoder_tokens: int,
        n_encoder_blocks: int = 6,
        n_decoder_blocks: int = 6,
        d_model: int = 512,
        n_heads: int = 8,
        dropout_p: float = 0.1,
        ffn_n_factors: int = 4,
        attn_bias: bool = True,
        ffn_bias: bool = True,
        out_proj_bias: bool = True,
        eps: float = 1e-5
    ) -> None:
        super().__init__()
        self.encoder = Encoder(
            n_tokens=n_encoder_tokens,
            n_blocks=n_encoder_blocks,
            d_model=d_model,
            n_heads=n_heads,
            dropout_p=dropout_p,
            ffn_n_factors=ffn_n_factors,
            attn_bias=attn_bias,
            ffn_bias=ffn_bias,
            eps=eps
        )
        self.decoder = Decoder(
            n_tokens=n_decoder_tokens,
            n_blocks=n_decoder_blocks,
            d_model=d_model,
            n_heads=n_heads,
            dropout_p=dropout_p,
            ffn_n_factors=ffn_n_factors,
            attn_bias=attn_bias,
            ffn_bias=ffn_bias,
            eps=eps
        )
        self.proj = nn.Linear(in_features=d_model, out_features=n_decoder_tokens, bias=out_proj_bias)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_mask: Optional[torch.Tensor] = None,
        y_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Encoder Handling
        x = self.encoder(x, x_mask)
        # Decoder Handling
        y = self.decoder(y, x, x_mask, y_mask)
        # Projection
        y = self.proj(y)
        return y