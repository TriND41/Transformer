import sys
sys.path.append('.')

import os
import torch
from tools.testing import Tester
from handlers.distribution import setup, cleanup
from typing import Literal, Union, Optional

def test(
    rank: Union[int, str, torch.device],
    world_size: int,
    test_path: str,
    checkpoint_path: str,
    num_testing_samples: Optional[int] = None,
    
) -> None:
    pass