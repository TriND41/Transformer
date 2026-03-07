import torch
import torch.nn as nn
from model.utils.block import DecoderBlock
from model.utils.position import PositionalEncoding
from model.utils.masking import extend_look_ahead_mask
from typing import Optional, Tuple

class Decoder(nn.Module):
    def __init__(
        self,
        num_tokens: int,
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
            DecoderBlock(d_model, num_heads, dropout_p, ffn_n_factors, attn_bias, ffn_bias, eps)
            for _ in range(num_blocks)
        ])
        self.proj = nn.Linear(d_model, num_tokens)

    @staticmethod
    def masking_setup(
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        context_mask: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, token_length, _ = x.size()
        if attn_mask is not None:
            if context_mask is not None:
                context_length = context_mask.size(1)

                context_mask = torch.logical_and(
                    attn_mask.unsqueeze(2).repeat([1, 1, context_length]),
                    context_mask.unsqueeze(1).repeat([1, token_length, 1])
                ).unsqueeze(1).logical_not()

            attn_mask = extend_look_ahead_mask(attn_mask).unsqueeze(1).logical_not()
        else:
            attn_mask = extend_look_ahead_mask(
                torch.ones([batch_size, token_length], dtype=torch.bool, device=x.device)
            ).unsqueeze(1).logical_not()

        return attn_mask, context_mask

    def get_kv_caches(self, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        k_caches = []
        v_caches = []
        for block in self.blocks:
            k, v = block.get_kv_cache(context)
            k_caches.append(k)
            v_caches.append(v)
        return torch.stack(k_caches, dim=0), torch.stack(v_caches, dim=0)
    
    def forward_kv_cache(
        self,
        x: torch.Tensor,
        k_caches: torch.Tensor,
        v_caches: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x += self.pe(x)
        attn_mask, context_mask = self.masking_setup(x, attn_mask, context_mask)
        for index, block in enumerate(self.blocks):
            x = block.forward_kv_cache(
                x, k_caches[index], v_caches[index],
                attn_mask, context_mask
            )
        x = self.proj(x[:, -1, :])
        return x

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x += self.pe(x)
        attn_mask, context_mask = self.masking_setup(x, attn_mask, context_mask)
        for block in self.blocks:
            x = block(x, context, attn_mask, context_mask)
        x = self.proj(x)
        return x