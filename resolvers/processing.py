import os
import numpy as np
from tokenizers import Tokenizer
from typing import List, Tuple

class TextProcessor:
    def __init__(self, tokenizer_path: str) -> None:
        assert os.path.exists(tokenizer_path)

        self.tokenizer = Tokenizer.from_file(tokenizer_path)

    def encode(self, text: str) -> np.ndarray:
        output = self.tokenizer.encode(text)
        return np.array(output.ids, dtype=np.int32)
    
    def process_tokens(self, seqs: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        lengths = []
        max_length = 0

        for seq in seqs:
            length = len(seq)
            lengths.append(length)
            if max_length < length:
                max_length = length

        padded_seqs = []
        for index, seq in enumerate(seqs):
            padded_seqs.append(
                np.pad(
                    array=seq,
                    pad_width=[0, max_length - lengths[index]],
                    constant_values=0.0
                )
            )

        return np.stack(padded_seqs, axis=0), np.array(lengths, dtype=np.int32)
    
    def __call__(self, texts: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        token_seqs = [self.encode(text) for text in texts]
        tokens, lengths = self.process_tokens(token_seqs)
        return tokens, lengths