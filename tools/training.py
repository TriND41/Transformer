import os
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.distributed as distributed
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler

from model.transformer import Transformer
from model.utils.masking import generate_padding_mask

from handlers.checkpoint import CheckpointManager, load_checkpoint
from handlers.configs import TransformerConfig
from handlers.constants import IGNORE_INDEX
from handlers.criterion import TransformerCriterion
from handlers.metric import MachineTranslationMetric
from handlers.symbols import TransformerCheckpointKey as CheckpointKey
from handlers.early_stopping import EarlyStopping
from handlers.logging import Logger
import handlers.gradient as gradient

import torchsummary
from tqdm import tqdm
from typing import Literal, Union, Tuple, Optional, Dict, Any

class Trainer:
    def __init__(
        self,
        rank: int,
        # Text
        encoder_tokenizer_path: str,
        decoder_tokenizer_path: Optional[str] = None,
        delim_token: str = "|",
        start_token: str = "<BOS>",
        end_token: str = "<EOS>",
        unk_token: str = "<UNK>",
        # Model
        
    ):
        pass