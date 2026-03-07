import torch
import torch.nn as nn
from model.modules.encoder import Encoder
from model.modules.decoder import Decoder
from typing import Optional, Tuple

class Transformer(nn.Module):
    def __init__(
        self,
        num_encoder_tokens: int,
        num_decoder_tokens: Optional[int] = None,
        num_encoder_blocks: int = 6,
        num_decoder_blocks: int = 6,
        d_model: int = 512,
        num_heads: int = 8,
        dropout_p: float = 0.0
    ) -> None:
        super().__init__()
        if num_decoder_tokens is None:
            self.share_embedding = True
            self.embedding = nn.Embedding(num_encoder_tokens, d_model)
        else:
            self.share_embedding = False
            self.encoder_embedding = nn.Embedding(num_encoder_tokens, d_model)
            self.decoder_embedding = nn.Embedding(num_decoder_tokens, d_model)

        self.encoder = Encoder(
            num_encoder_tokens, num_encoder_blocks,
            d_model, num_heads, dropout_p
        )
        self.decoder = Decoder(
            num_decoder_tokens, num_decoder_blocks,
            d_model, num_heads, dropout_p
        )

    def extract_context(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        x = self.encoder_embedding(x) if not self.share_embedding else self.embedding(x)
        x = self.encoder(x, attn_mask)
        return x
    
    def get_kv_caches(self, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        k_caches, v_caches = self.decoder.get_kv_caches(context)
        return k_caches, v_caches
    
    def embed_token(self, x: torch.Tensor) -> torch.Tensor:
        x = self.decoder_embedding(x) if not self.share_embedding else self.embedding(x)
        return x
    
    def extract_logits(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.decoder_embedding(x) if not self.share_embedding else self.embedding(x)
        x = self.decoder(x, context, attn_mask, context_mask)
        return x
    
    def decoding_step(
        self,
        x: torch.Tensor,
        k_caches: torch.Tensor,
        v_caches: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.decoder.forward_kv_cache(
            x, k_caches, v_caches,
            attn_mask, context_mask
        )
        return x

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_mask: Optional[torch.Tensor] = None,
        y_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.share_embedding:
            x = self.embedding(x)
            y = self.embedding(y)
        else:
            x = self.encoder_embedding(x)
            y = self.decoder_embedding(y)

        x = self.encoder(x, x_mask)
        y = self.decoder(y, x, y_mask, x_mask)
        return y