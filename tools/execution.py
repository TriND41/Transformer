import os
import torch
import numpy as np
from handlers.checkpoint import load_checkpoint
from model.transformer import Transformer
from model.utils.masking import generate_padding_mask
from model.modules.searching import GreedySearch
from resolvers.text import TextSentencePieceProcessor
from handlers.symbols import TransformerCheckpointKey as CheckpointKey
from typing import Union, Optional, Literal, Tuple, List

class Executor:
    def __init__(
        self,
        checkpoint_path: str,
        searching: Literal['greedy', 'beam'] = 'greedy',
        device: Union[str, int, torch.device] = 'cpu'
    ) -> None:
        assert os.path.exists(checkpoint_path)
        checkpoint = load_checkpoint(checkpoint_path)

        # Processing
        self.__shared_tokenizer = not CheckpointKey.DECODER_TEXT_PROCESSOR in checkpoint.keys()
        if self.__shared_tokenizer:
            self.text_proessor = TextSentencePieceProcessor(**checkpoint[CheckpointKey.TEXT_PROCESSOR])
            self.start_id = self.text_proessor.start_id
            self.end_id = self.text_proessor.end_id
        else:
            self.encoder_text_processor = TextSentencePieceProcessor(**checkpoint[CheckpointKey.ENCODER_TEXT_PROCESSOR])
            self.decoder_text_processor = TextSentencePieceProcessor(**checkpoint[CheckpointKey.DECODER_TEXT_PROCESSOR])
            self.start_id = self.decoder_text_processor.start_id
            self.end_id = self.decoder_text_processor.end_id

        # Modeling
        self.model = Transformer(**checkpoint[CheckpointKey.HYPER_PARAMS])
        self.model.load_state_dict(checkpoint[CheckpointKey.MODEL])
        self.model.eval()
        self.model.to(device)

        # Additional Modules
        self.searching_type = searching
        if searching == 'greedy':
            self.searching = GreedySearch(self.end_id)
        else:
            raise NotImplementedError()
        self.searching.to(device)
        self.searching.eval()

        # Information
        self.device = device

    @torch.no_grad()
    @torch.inference_mode()
    def extract_context(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        x = self.model.extract_context(x, attn_mask)
        return x
    
    @torch.no_grad()
    @torch.inference_mode()
    def embed_token(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model.embed_token(x)
        return x
    
    @torch.no_grad()
    @torch.inference_mode()
    def concatenate_embeddings(self, current: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
        return torch.concatenate([previous, current.unsqueeze(1)], dim=1)
    
    @torch.no_grad()
    @torch.inference_mode()
    def get_kv_caches(self, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        k_caches, v_caches = self.model.get_kv_caches(context)
        return k_caches, v_caches
    
    @torch.no_grad()
    @torch.inference_mode()
    def get_logits(
        self,
        token_embedding: torch.Tensor,
        k_caches: torch.Tensor,
        v_caches: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        logits = self.model.decoding_step(
            token_embedding, k_caches, v_caches,
            attn_mask, context_mask
        )
        return logits
    
    @torch.no_grad()
    @torch.inference_mode()
    def greedy_search(
        self,
        tokens: torch.Tensor,
        logits: torch.Tensor,
        completes: torch.Tensor,
        sum_logprobs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens, completes, sum_logprobs = self.searching(tokens, logits, completes, sum_logprobs)
        return tokens, completes, sum_logprobs
    
    @torch.no_grad()
    @torch.inference_mode()
    def generate_tokens(self, context: torch.Tensor, context_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        k_caches, v_caches = self.get_kv_caches(context)

        # Initialize states
        batch_size, max_decoder_steps, _ = context.size()

        tokens = torch.full([batch_size, 1], fill_value=self.start_id, dtype=torch.long, device=self.device)
        attn_mask = torch.ones([batch_size, 1], dtype=torch.bool, device=self.device)

        completes = torch.zeros([batch_size], dtype=torch.bool, device=self.device)
        sum_logprobs = torch.zeros([batch_size], dtype=torch.float, device=self.device)

        token_embedding: torch.Tensor

        # Auto-Regressive Stage
        num_steps = 0
        while not completes.all().item():
            if num_steps == 0:
                token_embedding = self.embed_token(tokens.squeeze(1)).unsqueeze(1)
            else:
                token_embedding = self.concatenate_embeddings(
                    self.embed_token(tokens[:, num_steps].contiguous()),
                    token_embedding
                )
            
            logits = self.get_logits(
                token_embedding, k_caches, v_caches,
                attn_mask, context_mask
            )
            tokens, completes, sum_logprobs = self.greedy_search(tokens, logits, completes, sum_logprobs)
            attn_mask = torch.concatenate([attn_mask, completes.logical_not().unsqueeze(1)], dim=1).contiguous()

            num_steps += 1
            if num_steps == max_decoder_steps:
                break

        return tokens, sum_logprobs.exp()
    
    @torch.no_grad()
    @torch.inference_mode()
    def translate(
        self,
        texts: List[str],
        parallel_decoding: bool = False,
        num_parallel_decoding_processes: Optional[int] = None
    ) -> Tuple[List[str], np.ndarray]:
        inputs, input_lengths = self.text_proessor(texts, auto_regressive=False)
        inputs = torch.tensor(inputs, dtype=torch.long, device=self.device)
        input_mask = generate_padding_mask(
            torch.tensor(input_lengths, dtype=torch.int32, device=self.device)
        )

        context = self.extract_context(inputs, input_mask)
        token_outputs, confident_scores = self.generate_tokens(context, input_mask)

        translations = self.text_proessor.batch_decode(
            sequences=token_outputs[:, 1:].cpu().numpy(),
            parallel_decoding=parallel_decoding,
            num_parallel_decoding_processes=num_parallel_decoding_processes
        )

        return translations, confident_scores.cpu().numpy()