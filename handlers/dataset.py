import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.csv as csv
from typing import Union, Optional, List, Tuple

class TransformerDataset(Dataset):
    def __init__(self, manifest: Union[str, pd.DataFrame, pa.Table], num_examples: Optional[int] = None) -> None:
        super().__init__()
        if isinstance(manifest, str):
            if '.parquet' in manifest:
                self.table = pq.read_table(manifest)
            elif '.csv' in manifest:
                self.table = csv.read_csv(manifest)
            elif '.tsv' in manifest:
                self.table = pa.Table.from_pandas(pd.read_csv(manifest, sep="\t"))
            else:
                raise("Invalid Manifest Format to read")
        elif isinstance(manifest, pd.DataFrame):
            self.table = pa.Table.from_pandas(manifest)
        else:
            self.table = manifest

        if num_examples is not None:
            self.table = self.table.slice(0, num_examples)