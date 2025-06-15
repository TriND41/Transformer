from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class TransformerConfig:
    n_encoder_tokens: int
    n_decoder_tokens: int
    n_encoder_blocks: int = 6
    n_decoder_blocks: int = 6
    d_model: int = 512
    n_heads: int = 8
    dropout_p: float = 0.0
    ffn_n_factors: int = 4
    attn_bias: bool = True
    ffn_bias: bool = True
    eps: float = 1e-5

    def update(self, configs: Dict[str, Any]) -> None:
        self.__dict__.update(configs)

@dataclass
class ProcessorConfig:
    encoder_tokenizer_path: str
    decoder_tokenizer_path: str
    start_token: str = "<BOS>"
    end_token: str = "<EOS>"
    unk_token: str = "<UNK>"

    def update(self, configs: Dict[str, Any]) -> None:
        self.__dict__.update(configs)