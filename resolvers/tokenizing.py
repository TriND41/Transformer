from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from handlers.common import load_configs
import logging
from typing import List, Union

logger = logging.getLogger(__name__)

def train_bpe_tokenizer(
    file_path: Union[str, List[str]],
    vocab_size: int,
    special_tokens_path: str,
    saved_path: str,
    min_frequency: int = 2
) -> None:
    if isinstance(file_path, str):
        file_path = [file_path]

    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = Whitespace()

    special_tokens = load_configs(special_tokens_path)

    trainer = BpeTrainer(vocab_size=vocab_size, min_frequency=min_frequency, special_tokens=special_tokens)
    trainer.train(file_path, trainer=trainer)

    tokenizer.save(saved_path)