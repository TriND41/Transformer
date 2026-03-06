from dataclasses import dataclass
from typing import Optional

@dataclass
class TransformerConfig:
    num_encoder_tokens: int
    num_decoder_tokens: Optional[int] = None
    num_encoder_blocks: int = 6
    num_decoder_blocks: int = 6
    d_model: int = 512
    num_heads: int = 8
    dropout_p: float = 0.1