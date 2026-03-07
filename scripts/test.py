import sys
sys.path.append('.')

import os
import torch
import torch.multiprocessing as mp
import pandas as pd
from tools.testing import Tester
from handlers.distribution import setup, cleanup
from typing import Literal, Union, Optional

def test(
    rank: Union[int, str, torch.device],
    world_size: int,
    test_path: str,
    checkpoint_path: str,
    num_testing_samples: Optional[int] = None,
    searching: Literal['greedy', 'beam'] = 'greedy',
    batch_size: int = 1,
    fp16: bool = False,
    parallel_decoding: bool = False,
    num_parallel_decoding_processes: Optional[int] = None
) -> None:
    try:
        if world_size > 1:
            setup(rank, world_size)
        
        tester = Tester(
            checkpoint_path, searching,
            rank
        )

        dataset = pd.read_csv(test_path)
        if num_testing_samples is not None:
            dataset = dataset[:num_testing_samples]

        tester.test(
            dataset, batch_size, fp16,
            parallel_decoding, num_parallel_decoding_processes
        )
    except Exception as e:
        raise ValueError(str(e))
    finally:
        if world_size > 1:
            cleanup()

def main(
    test_path: str,
    checkpoint_path: str,
    num_testing_samples: Optional[int] = None,
    searching: Literal['greedy', 'beam'] = 'greedy',
    batch_size: int = 1,
    fp16: bool = False,
    parallel_decoding: bool = False,
    num_parallel_decoding_processes: Optional[int] = None   
) -> None:
    assert os.path.exists(test_path)
    assert os.path.exists(checkpoint_path)

    num_gpus = torch.cuda.device_count()

    if num_gpus == 1 or num_gpus == 0:
        device = 'cuda' if num_gpus == 1 else 'cpu'
        test(
            device, num_gpus,
            test_path, checkpoint_path,
            num_testing_samples,
            searching, batch_size, fp16,
            parallel_decoding, num_parallel_decoding_processes
        )
    else:
        mp.spawn(
            fn=test,
            args=(
                 num_gpus,
                test_path, checkpoint_path,
                num_testing_samples,
                searching, batch_size, fp16,
                parallel_decoding, num_parallel_decoding_processes
            ),
            nprocs=num_gpus,
            join=True
        )

if __name__ == '__main__':
    import fire
    fire.Fire(main)