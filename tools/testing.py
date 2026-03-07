import os
import torch
import torch.distributed as distributed
import pandas as pd
import numpy as np
from tools.execution import Executor
from handlers.metric import MachineTranslationMetric
from handlers.distribution import sample_distributed_data
from handlers.mapping import MachineTranslationManifestKey as ManifestKey
from tqdm import tqdm
from typing import Union, Literal, Optional

class Tester:
    def __int__(
        self,
        checkpoint_path: str,
        searching: Literal['greedy', 'beam'] = 'greedy',
        device: Union[str, int, torch.device] = 'cpu'
    ) -> None:
        self.executor = Executor(checkpoint_path, searching, device)
        self.metric = MachineTranslationMetric()
        self.device = device
        self.device_type = 'cpu' if device == 'cpu' else 'cuda'

    def test(
        self,
        testset: pd.DataFrame,
        batch_size: int = 1,
        fp16: bool = False,
        parallel_decoding: bool = False,
        num_parallel_decoding_processes: Optional[int] = None
    ) -> None:
        if not distributed.is_initialized():
            _testset = testset
        else:
            _testset = sample_distributed_data(testset, rank=self.device, world_size=distributed.get_world_size())

        # Get Information
        source_texts = _testset[ManifestKey.SOURCE].tolist()
        labels = _testset[ManifestKey.TARGET].tolist()

        # Setup Batch
        num_items = len(_testset)
        num_batches = num_items // batch_size
        num_last_batch = num_items % batch_size
        if num_last_batch > 0:
            num_batches += 1
        else:
            num_last_batch = batch_size
        
        translations = []
        confident_scores = np.array([])
        for batch_idx in tqdm(range(num_batches)):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + (batch_size if batch_idx < num_batches - 1 else num_last_batch)

            # Setup batch input
            inputs = source_texts[start_idx: end_idx]

            # Inference
            with torch.autocast(device_type=self.device_type, enabled=fp16):
                outputs, probabilities = self.executor.translate(
                    texts=inputs,
                    parallel_decoding=parallel_decoding,
                    num_parallel_decoding_processes=num_parallel_decoding_processes
                )

            # Accumualata outputs
            translations += outputs
            confident_scores = np.concatenate([confident_scores, probabilities], axis=0)

        # Evaluate
        bleu_score = torch.tensor(
            self.metric.bleu_score(translations, labels),
            dtype=torch.float, device=self.device
        )
        confident_score = torch.tensor(confident_scores.mean(), dtype=torch.float, device=self.device)

        if distributed.is_initialized():
            distributed.all_reduce(bleu_score, op=distributed.ReduceOp.AVG)
            distributed.all_reduce(confident_score, op=distributed.ReduceOp.AVG)

        # Log information
        if self.device == 'cuda' or self.device == 0 or self.device == 'cpu':
            print(f"BLEU Score: {(bleu_score*100):.2f}%")
            print(f"Confident Score: {(confident_score*100):.2f}%")