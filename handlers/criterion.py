import torch
import torch.nn as nn
from typing import Optional

class TransformerCriterion(nn.Module):
    def __init__(self, ignore_index: int = -100) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.__cross_entropy_criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def cross_entropy_loss(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if padding_mask is not None:
            padding_mask = padding_mask.logical_not()
            outputs.masked_fill_(padding_mask.unsqueeze(2), value=self.ignore_index)
            targets.masked_fill_(padding_mask, value=self.ignore_index)

        loss = self.__cross_entropy_criterion(outputs.transpose(1, 2).float(), targets.long())
        return loss