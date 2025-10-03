import os
import numpy as np
from torch.utils.data import Dataset
from resolvers.processing import TextProcessor
from handlers.symbols import MachineTranslationManifestKey as ManifestKey
import pandas as pd
from typing import Optional, List, Tuple

class MachineTranslationDataset(Dataset):
    def __init__(self, manifest: str, src_text_processor: TextProcessor, dst_text_processor: TextProcessor, num_examples: Optional[int] = None) -> None:
        super().__init__()
        assert os.path.exists(manifest)
        self.table = pd.read_csv(manifest)
        if num_examples is not None:
            self.table = self.table[:num_examples]

        self.src_text_processor = src_text_processor
        self.dst_text_processor = dst_text_processor

    def __len__(self) -> int:
        return len(self.table)
    
    def __getitem__(self, index: int) -> Tuple[str, str]:
        index_df = self.table.iloc[index]

        src_text = index_df[ManifestKey.SRC]
        dst_text = index_df[ManifestKey.DST]

        return src_text, dst_text
    
    def collate(self, batch: List[Tuple[str, str]]) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
        src_texts, dst_texts = zip(*batch)

        src_ids, src_lengths = self.src_text_processor(src_texts)
        dst_ids, dst_lengths = self.dst_text_processor(dst_texts)

        targets = dst_ids[1:]

        return (src_ids, dst_ids, src_lengths, dst_lengths), targets