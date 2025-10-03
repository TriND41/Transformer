import torch
import torch.nn as nn
from model.utils.block import DecoderBlock
from model.utils.position import PositionalEncoding
from model.utils.common import extend_look_ahead_mask
from typing import Optional

class Decoder(nn.Module):
    def __init__(self, n_tokens: int, n_blocks: int, d_model: int, n_heads: int, dropout_p: float, ffn_n_factors: int = 4, attn_bias: bool = True, ffn_bias: bool = True, eps: float = 1e-5) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=n_tokens, embedding_dim=d_model)
        self.pe = PositionalEncoding(d_model)
        self.blocks = nn.ModuleList([
            DecoderBlock(d_model, n_heads, dropout_p, ffn_n_factors, attn_bias, ffn_bias, eps)
            for _ in range(n_blocks)
        ])

    def forward(self, x: torch.Tensor, context: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, context_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, length = x.size()

        x = self.embedding(x)
        x += self.pe(x)

        if attn_mask is not None and context_mask is not None:
            attn_length = attn_mask.size(1)
            context_length = context_mask.size(1)

            context_mask = torch.logical_and(
                attn_mask.unsqueeze(2).repeat([1, 1, context_length]),
                context_mask.unsqueeze(1).repeat([1, attn_length, 1])
            ).logical_not()

            attn_mask = extend_look_ahead_mask(attn_mask).unsqueeze(1).logical_not()
        elif attn_mask is not None:
            attn_mask = extend_look_ahead_mask(attn_mask).unsqueeze(1).logical_not()
        else:
            attn_mask = extend_look_ahead_mask(
                torch.ones([batch_size, length], dtype=torch.bool, device=x.device)
            )

        for block in self.blocks:
            x = block(x, context, attn_mask, context_mask)

        return x