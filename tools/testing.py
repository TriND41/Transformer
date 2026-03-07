import os
import torch
import torch.distributed as distributed
import pandas as pd
import numpy as np
from tools.execution import Executor
from typing import Union, Literal

class Tester:
    def __int__(
        self,
        checkpoint_path: str,
        searching: Literal['greedy', 'beam'] = 'greedy',
        device: Union[str, int, torch.device] = 'cpu'
    ) -> None:
        self.executor = Executor(checkpoint_path, searching, device)
        self.device = device
        self.device_type = 'cpu' if device == 'cpu' else 'cuda'

    def test(
        self,
        testset: pd.DataFrame,
        batch_size: int = 1,
        fp16: bool = False
    ) -> None:
        pass