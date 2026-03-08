import os
import torch
import torch.distributed as distributed
import numpy as np
import pandas as pd
from tools.execution import Executor
from handlers.distribution import sample_distributed_data
from tqdm import tqdm
from typing import Literal, Optional, Union, List

class Runner:
    def __init__(
        self,
        checkpoint_path: str,
        searching: Literal['greedy', 'beam'] = 'greedy',
        device: Union[str, int, torch.device] = 'cpu'
    ) -> None:
        self.executor = Executor(
            checkpoint_path,
            searching, device
        )

        self.device = device
        self.device_type = 'cpu' if device == 'cpu' else 'cuda'

    def run(
        self,
        texts: List[str],
        batch_size: int = 1,
        fp16: bool = False,
        parallel_decoding: bool = False,
        num_parallel_decoding_processes: Optional[int] = None,
        output_result_path: str = "./results.csv"
    ) -> None:
        if not distributed.is_initialized():
            _texts = texts
        else:
            _texts = sample_distributed_data(texts, rank=self.device, world_size=distributed.get_world_size())

        num_items = len(_texts)
        num_batches = num_items // batch_size
        num_last_batch = num_items % batch_size
        if num_last_batch > 0:
            num_batches += 1
        else:
            num_last_batch = batch_size

        translations = []
        confident_scores = np.array([])
        for batch_idx in tqdm(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + (batch_size if batch_idx < num_batches - 1 else num_last_batch)

            inputs = _texts[start_idx: end_idx]
            with torch.autocast(device_type=self.device_type, enabled=fp16):
                outputs, probabilities = self.executor.translate(inputs, parallel_decoding, num_parallel_decoding_processes)

            translations += outputs
            confident_scores = np.concatenate([confident_scores, probabilities], axis=0)

        confident_score = torch.tensor(confident_scores.mean(), dtype=torch.float, device=self.device)
        if distributed.is_initialized():
            distributed.all_reduce(confident_score, op=distributed.ReduceOp.AVG)

        if self.device == 'cuda' or self.device == 0 or self.device == 'cpu':
            print(f"Mean Confident Score: {(confident_score*100):.2f}%")

            if distributed.is_initialized():
                _translations = [None for _ in range(distributed.get_world_size())]
                _confident_scores = [None for _ in range(distributed.get_world_size())]

                distributed.all_gather_object(_translations, translations)
                distributed.all_gather_object(_confident_scores, confident_scores)

                translations = [item for sublist in _translations for item in sublist]
                confident_scores = np.concatenate(_confident_scores, axis=0)

            df = pd.DataFrame({
                'source_text': texts,
                'prediction': translations,
                'confident_score': confident_scores
            })
            df.to_csv(output_result_path, index=False)