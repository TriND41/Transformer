import os

import torch
import torch.distributed as distributed

from model.transformer import Transformer

from handlers.checkpoint import CheckpointManager, load_checkpoint
from handlers.configs import TransformerConfig, SrcTextProcessorConfig, DstTextProcessorConfig

from typing import Optional, Dict, Any

class TransformerTrainer:
    def __init__(
        self,
        rank: int,
        # Text configs
        src_tokenizer_path: str,
        dst_tokenizer_path: str,
        # Model Configs
        n_encoder_blocks: int = 6,
        n_decoder_blocks: int = 6,
        d_model: int = 512,
        n_heads: int = 8,
        dropout_p: float = 0.1,
        # Optimizer Configs
        lr: Optional[float] = None,
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
        
            