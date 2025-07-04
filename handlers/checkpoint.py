import os
import torch
import torch.distributed as distributed
from torch.nn import Module
from torch.optim import Optimizer

from utils.dictionaries import CheckpointKey
from handlers.constants import CHECKPOINT_EXTENSION

from collections import OrderedDict
from typing import Dict, Union, Any

def is_ddp_checkpoint(state_dict: OrderedDict):
    for key in state_dict.keys():
        if key.startswith("module.") == False:
            return False
    return True

def change_format_single_gpu(dict: OrderedDict) -> OrderedDict:
    new_dict = OrderedDict()
    for key, value in dict.items():
        new_key = key.replace("module.", '', 1)
        new_dict[new_key] = value
    return new_dict

def change_format_multi_gpus(dict: OrderedDict) -> OrderedDict:
    new_dict = OrderedDict()
    for key, value in dict.items():
        new_key = f"module.{key}"
        new_dict[new_key] = value
    return new_dict

def load_model(state_dict: Union[str, OrderedDict], model: Module) -> None:
    if isinstance(state_dict, str):
        state_dict = torch.load(state_dict, map_location='cpu')['model']
    is_ddp = is_ddp_checkpoint(state_dict)
    init_ddp = distributed.is_initialized()
    if init_ddp == False and is_ddp:
        state_dict = change_format_single_gpu(state_dict)
    elif init_ddp and is_ddp == False:
        state_dict = change_format_multi_gpus(state_dict)
    model.load_state_dict(state_dict)

def load_checkpoint(path: str) -> Dict[str, Any]:
    return torch.load(path, map_location='cpu', weights_only=False)

class CheckpointManager:
    def __init__(self, saved_folder: str, n_savings: int = 3) -> None:
        if os.path.exists(saved_folder) == False:
            os.makedirs(saved_folder)

        self.saved_folder = saved_folder
        self.n_savings = n_savings
        self.saved_checkpoints = []

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        return load_checkpoint(path)
    
    def is_full_checkpoints(self) -> bool:
        return len(self.saved_checkpoints) >= self.n_savings
    
    def remove_first(self) -> None:
        os.remove(f"{self.saved_folder}/{self.saved_checkpoints[0]}.{CHECKPOINT_EXTENSION}")
        self.saved_checkpoints.pop(0)

    def __save_checkpoint(self, checkpoint: Dict[str, Any], iterations: int) -> None:
        torch.save(checkpoint, f"{self.saved_folder}/{iterations}.{CHECKPOINT_EXTENSION}")
        self.saved_checkpoints.append(iterations)

    def save_checkpoint(
        self,
        model: Module,
        optimizer: Optimizer,
        n_steps: int,
        n_epochs: int,
        model_config: Dict[str, Any],
        processor_config: Dict[str, Any],
        logging: bool = False
    ) -> None:
        checkpoint = {
            CheckpointKey.MODEL: model.state_dict(),
            CheckpointKey.OPTIMIZER: optimizer.state_dict(),
            CheckpointKey.N_ITERATIONS: n_steps,
            CheckpointKey.N_EPOCHS: n_epochs,
            CheckpointKey.MODEL_CONFIG: model_config,
            CheckpointKey.PROCESSOR_CONFIG: processor_config
        }

        if self.is_full_checkpoints():
            self.remove_first()

        self.__save_checkpoint(checkpoint, n_steps)

        if logging:
            print(f"Saved Checkpoint at {self.saved_folder}/{n_steps}.{CHECKPOINT_EXTENSION} - Iteration {n_steps} - Epoch {n_epochs}")