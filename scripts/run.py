import sys
sys.path.append('.')

import os
import torch
from tools.execution import Executor
from typing import Literal, Union, Optional

def run(
    checkpoint_path: str,
    text: str,
    fp16: bool = False,
    searching: Literal['greedy', 'beam'] = 'greedy',
    parallel_decoding: bool = False,
    num_parallel_decoding_processes: Optional[int] = None,
    device: Union[str, int, torch.device] = 'cpu'
) -> None:
    assert os.path.exists(checkpoint_path)
    
    executor = Executor(checkpoint_path, searching=searching, device=device)

    device_type = 'cpu' if device == 'cpu' else 'cuda'
    with torch.autocast(device_type=device_type, enabled=fp16):
        predictions, probabilities = executor.translate(
            [text],
            parallel_decoding=parallel_decoding,
            num_parallel_decoding_processes=num_parallel_decoding_processes
        )
    
    print(f"Translation: {predictions[0]}")
    print(f"Confident Score: {(probabilities[0]*100):.2f}%")

if __name__ == '__main__':
    import fire
    fire.Fire(run)
