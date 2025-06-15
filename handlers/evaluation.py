import torch
import torch.nn as nn

class TransformerCriterion(nn.Module):
    def __init__(self, ignore_index: int = -100) -> None:
        super().__init__()
        self.cross_entropy_criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def cross_entropy_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.cross_entropy_criterion(outputs.transpose(1, 2).float(), targets.float())