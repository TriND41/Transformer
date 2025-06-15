import os

import torch
import torch.distributed as distributed

from model.transformer import Transformer
from collections import OrderedDict
from typing import Optional, Dict, Any

class TransformerTrainer:
    def __init__(
        self,
        rank: int,
        # Model Configs
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
        eps: float = 1e-5,
        # Optimizer Configs

        # Checkpoint Configs
        checkpoint_path: Optional[str] = None,
        checkpoint_folder: str = "./checkpoints",
        n_saved_checkpoints: int = 3,
        save_checkpoint_after_epochs: int = 1
    ) -> None:
        self.rank = rank

        checkpoint: Optional[Dict[str, Any]] = None
        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        