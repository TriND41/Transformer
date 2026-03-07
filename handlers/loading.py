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
        source_text_processor: TextProcessor,
        target_text_processor: Optional[TextProcessor] = None,
        num_examples: Optional[int] = None
    ) -> None:
        super().__init__()
        self.table = pd.read_csv(manifest)
        if num_examples is not None:
            self.table = self.table[:num_examples]

        if target_text_processor is not None:
            self.source_text_processor = source_text_processor
            self.target_text_processor = target_text_processor
            self.__shared_tokenizer = False
        else:
            self.text_processor = source_text_processor
            self.__shared_tokenizer = True

    def __len__(self) -> int:
        return len(self.table)
    
    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        index_df = self.table.iloc[index]

        source_text = index_df[ManifestKey.SOURCE]
        target_text = index_df[ManifestKey.TARGET]

        if not self.__shared_tokenizer:
            source_tokens = self.source_text_processor.encode(source_text)
            target_tokens = self.target_text_processor.encode(target_text)
        else:
            source_tokens = self.text_processor.encode(source_text)
            target_tokens = self.text_processor.encode(target_text)

        return source_tokens, target_tokens 