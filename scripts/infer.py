import sys
sys.path.append('.')

import os
import torch
import torch.multiprocessing as mp
import io
from tools.running import Runner
from handlers.distribution import setup, cleanup
from handlers.common import mkdir_root_folder
from typing import Literal, Union, Optional

def infer(
    rank: Union[str, int, torch.device],
    world_size: int,
    data_path: str,
    checkpoint_path: str,
    num_samples: Optional[int] = None,
    batch_size: int = 1,
    fp16: bool = False,
    searching: Literal['greedy', 'beam'] = 'greedy',
    parallel_decoding: bool = False,
    num_parallel_decoding_processes: Optional[int] = None,
    output_result_path: str = "./results"
) -> None:
    try:
        if world_size > 1:
            setup(rank, world_size)

        runner = Runner(
            checkpoint_path,
            searching, rank
        )

        texts = io.open(data_path, encoding='utf8').read().strip().split("\n")
        if num_samples is not None:
            texts = texts[:num_samples]

        runner.run(
            texts=texts,
            batch_size=batch_size,
            fp16=fp16,
            parallel_decoding=parallel_decoding,
            num_parallel_decoding_processes=num_parallel_decoding_processes,
            output_result_path=output_result_path
        )
    except Exception as e:
        raise ValueError(str(e))
    finally:
        if world_size > 1:
            cleanup()

def main(
    data_path: str,
    checkpoint_path: str,
    num_samples: Optional[int] = None,
    batch_size: int = 1,
    fp16: bool = False,
    searching: Literal['greedy', 'beam'] = 'greedy',
    parallel_decoding: bool = False,
    num_parallel_decoding_processes: Optional[int] = None,
    output_result_path: str = "./results"
) -> None:
    assert os.path.exists(data_path)
    assert os.path.exists(checkpoint_path)

    mkdir_root_folder(output_result_path)
    num_gpus = torch.cuda.device_count()

    if num_gpus == 1 or num_gpus == 0:
        device = 'cuda' if num_gpus == 1 else 'cpu'
        infer(
            device, num_gpus,
            data_path, checkpoint_path,
            num_samples, batch_size, fp16,
            searching, parallel_decoding, num_parallel_decoding_processes,
            output_result_path
        )
    else:
        mp.spawn(
            fn=infer,
            args=(
                num_gpus,
                data_path, checkpoint_path,
                num_samples, batch_size, fp16,
                searching, parallel_decoding, num_parallel_decoding_processes,
                output_result_path
            ),
            nprocs=num_gpus,
            join=True
        )

if __name__ == '__main__':
    import fire
    fire.Fire(main)