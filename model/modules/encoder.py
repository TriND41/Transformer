import torch
import torch.nn as nn
from model.utils.block import EncoderBlock
from model.utils.position import PositionalEncoding
from typing import Optional, List, Tuple

class Encoder(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        d_model: int,
        num_heads: int,
        dropout_p: float,
        ffn_n_factors: int = 4,
        attn_bias: bool = True,
        ffn_bias: bool = True,
        eps: float = 1e-5
    ) -> None:
        super().__init__()
        self.pe = PositionalEncoding(d_model)
        self.blocks = nn.ModuleList([
            EncoderBlock(d_model, num_heads, dropout_p, ffn_n_factors, attn_bias, ffn_bias, eps)
            for _ in range(num_blocks)
        ])

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        x += self.pe(x)
        
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(2).logical_not()
        
        for block in self.blocks:
            x = block(x, attn_mask)

        return x