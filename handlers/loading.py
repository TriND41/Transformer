import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from resolvers.text import TextProcessor
from handlers.mapping import MachineTranslationManifestKey as ManifestKey
from typing import Tuple, Optional

class MachineTranslationDataset(Dataset):
    def __init__(
        self,
        manifest: str,
        text_processor: TextProcessor,
        num_examples: Optional[int] = None
    ) -> None:
        super().__init__()
        self.table = pd.read_csv(manifest)
        if num_examples is not None:
            self.table = self.table[:num_examples]

        self.text_processor = text_processor

    def __len__(self) -> int:
        return len(self.table)
    
    def __getitem__(self, index: int) -> Tuple[str, str]:
        index_df = self.table.iloc[index]

        src_text = index_df[ManifestKey.SOURCE]
        targ_text = index_df[ManifestKey.TARGET]

        return src_text, targ_text