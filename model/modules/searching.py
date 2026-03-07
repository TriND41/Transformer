import torch
import torch.nn as nn
from typing import Tuple

class GreedySearch(nn.Module):
    def __init__(self, end_id: int) -> None:
        super().__init__()
        self.end_id = end_id
    
    def decode(
        self,
        tokens: torch.Tensor,
        logits: torch.Tensor,
        completes: torch.Tensor,
        sum_logprobs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Find next tokens by the max probability
        logits, next_tokens = torch.max(logits, dim=-1)
        next_tokens = next_tokens.type(tokens.dtype)
        
        # Accumulate log of probability
        sum_logprobs += logits * completes.logical_not()

        # Concat the next token to the previous ones
        next_tokens[completes] = self.end_id
        completes = torch.logical_or(completes, (next_tokens == self.end_id))
        tokens = torch.concatenate([tokens, next_tokens.unsqueeze(dim=-1)], dim=-1) # [batch_size, length + 1]

        return tokens, completes, sum_logprobs

    def forward(
        self,
        tokens: torch.Tensor,
        logits: torch.Tensor,
        completes: torch.Tensor,
        sum_logprobs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = torch.log_softmax(logits, dim=1)
        tokens, completes, sum_logprobs = self.decode(tokens, logits, completes, sum_logprobs)
        return tokens, completes, sum_logprobs
