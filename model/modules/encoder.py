import torch
import torch.nn as nn
from model.utils.block import EncoderBlock
from model.utils.position import PositionalEncoding
from typing import Optional, Union, List, Tuple

class Encoder(nn.Module):
    def __init__(self, n_tokens: int, n_blocks: int, d_model: int, n_heads: int, dropout_p: float, ffn_n_factors: int = 4, attn_bias: bool = True, ffn_bias: bool = True, eps: float = 1e-5) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=n_tokens, embedding_dim=d_model)
        self.pe = PositionalEncoding(d_model)
        self.blocks = nn.ModuleList([
            EncoderBlock(d_model, n_heads, dropout_p, ffn_n_factors, attn_bias, ffn_bias, eps)
            for _ in range(n_blocks)
        ])

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, get_weights: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        x = self.embedding(x)
        x += self.pe(x)

        weights = []
        if attn_mask is not None:
            attn_mask.unsqueeze_(1).unsqueeze_(2).logical_not_()
        for block in self.blocks:
            x, block_weights = block(x, attn_mask, get_weights=get_weights)
            weights.append(block_weights)

        if not get_weights:
            return x
        else:
            return x, weights