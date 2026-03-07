import os
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.distributed as distributed
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, RandomSampler, DistributedSampler

from model.transformer import Transformer
from model.utils.masking import generate_padding_mask
from model.modules.searching import GreedySearch
from resolvers.text import TextSentencePieceProcessor

from handlers.checkpoint import CheckpointManager, load_checkpoint
from handlers.configs import TransformerConfig
from handlers.dictionary import TextProcessorConfig
from handlers.constants import IGNORE_INDEX
from handlers.criterion import TransformerCriterion
from handlers.metric import MachineTranslationMetric
from handlers.symbols import TransformerCheckpointKey as CheckpointKey
from handlers.early_stopping import EarlyStopping
from handlers.loading import MachineTranslationDataset
from handlers.logging import Logger
import handlers.gradient as gradient

import numpy as np
import torchsummary
from tqdm import tqdm
from typing import Literal, Tuple, Optional, Dict, Any, List, Callable

def configure_dataloader(
    dataset: Dataset,
    collate_fn: Callable[[Any], Any],
    batch_size: int = 1
) -> DataLoader:
    if not distributed.is_initialized():
        sampler = RandomSampler(dataset)
    else:
        sampler = DistributedSampler(dataset, num_replicas=distributed.get_world_size(), rank=distributed.get_rank())
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate_fn
    )
    return dataloader

class Trainer:
    def __init__(
        self,
        rank: int,
        # Text
        encoder_tokenizer_path: str,
        decoder_tokenizer_path: Optional[str] = None,
        delim_token: str = "|",
        start_token: str = "<BOS>",
        end_token: str = "<EOS>",
        unk_token: str = "<UNK>",
        # Model
        num_encoder_blocks: int = 6,
        num_decoder_blocks: int = 6,
        d_model: int = 512,
        num_heads: int = 8,
        dropout_p: float = 0.1,
        # Learning
        lr: Optional[float] = None,
        update_lr: bool = False,
        # Checkpoint
        checkpoint_path: Optional[str] = None,
        checkpoint_folder: str = "./checkpoints",
        num_saved_checkpoints: int = 3,
        save_checkpoint_after_steps: Optional[int] = None,
        save_checkpoint_after_epochs: int = 1,
        # Early stopping
        early_stopping: bool = False,
        num_patiences: int = 3,
        observe: Literal['loss', 'score'] = 'loss',
        score_type: Literal['down', 'up'] = 'down',
        # Logging
        logging: bool = False,
        logging_project: str = "Transformer - Machine Translation",
        logging_name: Optional[str] = None
    ) -> None:
        self.rank = rank
        self.num_steps, self.num_epochs = 0, 0

        checkpoint: Optional[Dict[str, Any]] = None
        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            checkpoint = load_checkpoint(checkpoint_path)

            self.hyper_params = TransformerConfig(**checkpoint[CheckpointKey.HYPER_PARAMS])
            if CheckpointKey.DECODER_TEXT_PROCESSOR in checkpoint.keys():
                self.__shared_tokenizer = False
                self.encoder_text_configs = TextProcessorConfig(**checkpoint[CheckpointKey.ENCODER_TEXT_PROCESSOR])
                self.decoder_text_configs = TextProcessorConfig(**checkpoint[CheckpointKey.DECODER_TEXT_PROCESSOR])
            else:
                self.__shared_tokenizer = True
                self.text_configs = TextProcessorConfig(**checkpoint[CheckpointKey.TEXT_PROCESSOR])
        else:
            if decoder_tokenizer_path is not None and os.path.exists(decoder_tokenizer_path):
                self.__shared_tokenizer = False
                self.encoder_text_configs = TextProcessorConfig(
                    encoder_tokenizer_path,
                    delim_token, start_token, end_token, unk_token
                )
                self.decoder_text_configs = TextProcessorConfig(
                    decoder_tokenizer_path,
                    delim_token, start_token, end_token, unk_token
                )
            else:
                self.__shared_tokenizer = True
                self.text_configs = TextProcessorConfig(
                    encoder_tokenizer_path,
                    delim_token, start_token, end_token, unk_token
                )

        # Processing
        if self.__shared_tokenizer:
            self.encoder_text_processor = TextSentencePieceProcessor(**self.encoder_text_configs.__dict__)
            self.decoder_text_processor = TextSentencePieceProcessor(**self.decoder_text_configs.__dict__)
        else:
            self.text_processor = TextSentencePieceProcessor(**self.text_configs.__dict__)

        if checkpoint is None:
            if not self.__shared_tokenizer:
                num_encoder_tokens = self.encoder_text_processor.num_vocabs
                num_decoder_tokens = self.decoder_text_processor.num_vocabs
            else:
                num_encoder_tokens = self.text_processor.num_vocabs
                num_decoder_tokens = None

            self.hyper_params = TransformerConfig(
                num_encoder_tokens, num_decoder_tokens,
                num_encoder_blocks, num_decoder_blocks,
                d_model, num_heads, dropout_p
            )

        # Modeling
        self.model = Transformer(**self.hyper_params.__dict__)
        self.model.to(rank)
        if distributed.is_initialized():
            self.model = DDP(self.model, device_ids=[rank])

        # Optimization
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr if lr is not None else 3e-4)
        self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99974)

        # Additional module
        end_id = self.decoder_text_processor.end_id if not self.__shared_tokenizer else self.text_processor.end_id
        self.greedy_searching = GreedySearch(end_id)
        self.greedy_searching.to(rank)
        self.greedy_searching.eval()

        # Evaluation
        self.criterion = TransformerCriterion(ignore_index=IGNORE_INDEX)
        self.criterion.to(rank)

        self.metric = MachineTranslationMetric()

        # Load Weights
        if checkpoint is not None:
            self.__load_weights(checkpoint)

        # Apply learning rate directly
        if lr is not None and update_lr:
            self.optimizer.param_groups[0]['lr'] = lr

        self.start_id = self.decoder_text_processor.start_id if not self.__shared_tokenizer else self.text_processor.start_id

        # Informative Modules
        if self.rank == 0:
            # Checkpoint
            self.checkpoint_manager = CheckpointManager(checkpoint_folder, num_saved_checkpoints)
            self.save_checkpoint_after_steps = save_checkpoint_after_steps
            self.save_checkpoint_after_epochs = save_checkpoint_after_epochs

            # Early stopping
            self.early_stopping: Optional[EarlyStopping] = None
            if early_stopping:
                self.early_stopping = EarlyStopping(num_patiences, score_type)
                self.observe = observe

            # Logging
            self.logger: Optional[Logger] = None
            if logging:
                self.logger = Logger(logging_project, logging_name)

            # Summary
            print("\nModel Summary:")
            torchsummary.summary(self.model)

    def get_source_tokenizer(self) -> TextSentencePieceProcessor:
        if self.__shared_tokenizer:
            return self.text_processor
        else:
            return self.encoder_text_processor
        
    def get_target_tokenizer(self) -> Optional[TextSentencePieceProcessor]:
        if self.__shared_tokenizer:
            return None
        else:
            return self.decoder_text_processor

    def __load_weights(self, checkpoint: Dict[str, Any]) -> None:
        self.model.load_state_dict(checkpoint[CheckpointKey.MODEL])
        self.optimizer.load_state_dict(checkpoint[CheckpointKey.OPTIMIZER])
        self.scheduler.load_state_dict(checkpoint[CheckpointKey.SCHEDULER])
        self.num_steps = checkpoint[CheckpointKey.ITERATION]
        self.num_epochs = checkpoint[CheckpointKey.EPOCH]

    def __save_checkpoint(self, logging: bool = False) -> None:
        checkpoint = {
            # Hyper params
            CheckpointKey.HYPER_PARAMS: self.hyper_params.__dict__,
            # Weights
            CheckpointKey.MODEL: self.model.state_dict(),
            CheckpointKey.OPTIMIZER: self.optimizer.state_dict(),
            CheckpointKey.SCHEDULER: self.scheduler.state_dict(),
            # Information
            CheckpointKey.ITERATION: self.num_steps,
            CheckpointKey.EPOCH: self.num_epochs
        }

        if self.__shared_tokenizer:
            checkpoint[CheckpointKey.TEXT_PROCESSOR] = self.text_configs.__dict__
        else:
            checkpoint[CheckpointKey.ENCODER_TEXT_PROCESSOR] = self.encoder_text_configs.__dict__
            checkpoint[CheckpointKey.DECODER_TEXT_PROCESSOR] = self.decoder_text_configs.__dict__

        self.checkpoint_manager.save_checkpoint(checkpoint, self.num_epochs, self.num_steps, logging=logging)

    @torch.no_grad()
    def __collate(
        self,
        batch: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        # Get Data
        source_token_sentences, target_token_sentences = zip(*batch)
        
        # Setup Batch
        if not self.__shared_tokenizer:
            source_token_sequences, source_token_lengths = self.encoder_text_processor(source_token_sentences, auto_regressive=False)
            output_token_sequences, output_token_lengths = self.decoder_text_processor(target_token_sentences, auto_regressive=True)
        else:
            source_token_sequences, source_token_lengths = self.encoder_text_processor(source_token_sentences, auto_regressive=False)
            output_token_sequences, output_token_lengths = self.decoder_text_processor(target_token_sentences, auto_regressive=True)

        target_token_sequences = output_token_sequences[:, 1:]

        # Convert to Tensor
        source_token_sequences = torch.tensor(source_token_sequences, dtype=torch.long, device=self.rank)
        source_token_lengths = torch.tensor(source_token_lengths, dtype=torch.int32, device=self.rank)

        output_token_sequences = torch.tensor(output_token_sequences, dtype=torch.long, device=self.rank)
        output_token_lengths = torch.tensor(output_token_lengths, dtype=torch.int32, device=self.rank)

        target_token_sequences = torch.tensor(target_token_sequences, dtype=torch.long, device=self.rank)
        
        return ((source_token_sequences, source_token_lengths), (output_token_sequences, output_token_lengths)), target_token_sequences
    
    def __generate_tokens(self, context: torch.Tensor, context_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        k_caches, v_caches = self.model.get_kv_caches(context)

        # Initialize States
        batch_size, max_decoder_steps, _ = context.size()

        tokens = torch.full([batch_size, 1], fill_value=self.start_id, dtype=torch.long, device=self.rank)
        attn_mask = torch.ones([batch_size, 1], dtype=torch.bool, device=self.rank)

        completes = torch.zeros([batch_size], dtype=torch.bool, device=self.rank)
        sum_logprobs = torch.zeros([batch_size], dtype=torch.float, device=self.rank)

        token_embedding: torch.Tensor

        # Auto-Regressive Stage
        num_steps = 0
        while not completes.all().item():
            if num_steps == 0:
                token_embedding = self.model.embed_token(tokens.squeeze(1)).unsqueeze(1)
            else:
                token_embedding = torch.concatenate([
                    self.model.embed_token(tokens[:, num_steps].contiguous()),
                    token_embedding
                ], dim=1).contiguous()

            # Get logits
            logits = self.model.decoding_step(
                token_embedding, k_caches, v_caches,
                attn_mask, context_mask
            )

            tokens, completes, sum_logprobs = self.greedy_searching.decode(
                tokens, logits.log_softmax(dim=1),
                completes, sum_logprobs
            )

            attn_mask = torch.concatenate([attn_mask, completes.logical_not().unsqueeze(1)], dim=1).contiguous()

            # Stop Criterion
            num_steps += 1
            if num_steps == max_decoder_steps:
                break
        
        return tokens, sum_logprobs.exp()
    
    def __validate(self, dataloader: DataLoader, fp16: bool = False) -> None:
        self.model.eval()
        val_loss = 0.0
        val_bleu_score = 0.0
        val_confident_score = 0.0

        with torch.no_grad():
            for ((inputs, input_lengths), (sequences, sequence_lengths)), targets in tqdm(dataloader, leave=False):
                input_mask = generate_padding_mask(input_lengths)
                sequence_mask = generate_padding_mask(sequence_lengths)

                with torch.autocast(device_type='cuda', enabled=fp16):
                    context = self.model.extract_context(inputs, input_mask)
                    outputs = self.model.extract_logits(sequences, context, sequence_mask, input_mask)
                    tokens, probabilities = self.__generate_tokens(context, input_mask)

                    predictions = self.text_processor.batch_decode(tokens[:, 1:].cpu().numpy())
                    labels = self.text_processor.batch_decode(targets.cpu().numpy())

                    with torch.autocast(device_type='cuda', enabled=False):
                        loss = self.criterion.cross_entropy_loss(outputs, targets, input_mask[:, 1:])
                        bleu_score = self.metric.bleu_score(predictions, labels)

            val_loss += loss
            val_bleu_score += torch.tensor(bleu_score, dtype=torch.float, device=self.rank)
            val_confident_score += probabilities.mean()

        if self.rank == 0:
            print(f"Validation Loss: {val_loss:.4f}")
            print(f"Validation BLEU Score: {(val_bleu_score*100):.2f}%")
            print("----------------------------------")
            print(f"Validation Confident Score: {(val_confident_score*100):.2f}%")

            if self.logger is not None:
                self.logger.log({
                    'val_loss': val_loss,
                    'val_bleu_score': val_bleu_score,
                    'val_confident_score': val_confident_score
                }, self.num_steps)

    def train(
        self,
        dataset: MachineTranslationDataset,
        batch_size: int = 1,
        num_epochs: int = 1,
        # Validation
        val_dataset: Optional[MachineTranslationDataset] = None,
        val_batch_size: int = 1,
        # Addition
        fp16: bool = False,
        gradient_clipping: bool = False,
        clipping_value: Optional[float] = None
    ) -> None:
        dataloader = configure_dataloader(dataset, collate_fn=self.__collate, batch_size=batch_size)
        val_dataloader = configure_dataloader(val_dataset, collate_fn=self.__collate, batch_size=val_batch_size) if val_dataset is not None else None

        scaler = torch.GradScaler(enabled=fp16)
        for epoch in range(num_epochs):
            if distributed.is_initialized():
                dataloader.sampler.set_epoch(self.num_epochs)

            if self.rank == 0:
                print(f"\Epoch {epoch+1}/{num_epochs}\n====================")

            self.model.train()
            train_loss = 0.0
            train_gradient_norm = 0.0
            for ((inputs, input_lengths), (sequences, sequence_lengths)), targets in tqdm(dataloader):
                with torch.no_grad():
                    input_mask = generate_padding_mask(input_lengths)
                    sequence_mask = generate_padding_mask(sequence_lengths)

                with torch.autocast(device_type='cuda', enabled=fp16):
                    outputs = self.model(inputs, sequences, input_mask, sequence_mask)[:, :-1, :]
                    with torch.autocast(device_type='cuda', enabled=False):
                        loss = self.criterion.cross_entropy_loss(outputs, targets, sequence_mask[:, 1:])
                        assert torch.isnan(loss) == False
                
                # Backward
                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                gradient_norm = gradient.compute_gradient_norm(
                    self.model.parameters(),
                    clipping=gradient_clipping, clipping_value=clipping_value
                )
                scaler.unscale_(self.optimizer)
                scaler.step(self.optimizer)

                # Update iteration information
                self.num_steps += 1
                scaler.update()
                train_loss += loss
                train_gradient_norm += gradient_norm

            # Update epoch information
            self.num_epochs += 1

            train_loss /= len(dataloader)
            train_gradient_norm /= len(dataloader)
            if distributed.is_initialized():
                distributed.all_reduce(train_loss, op=distributed.ReduceOp.AVG)
                distributed.all_reduce(train_gradient_norm, op=distributed.ReduceOp.AVG)

            # Log information
            if self.rank == 0:
                current_lr = self.optimizer.param_groups[0]['lr']

                print(f"Training Loss: {train_loss:.4f}")
                print(f"Gradient Norm: {train_gradient_norm:.4f}")
                print(f"Learning Rate: {current_lr}")

                if epoch % self.save_checkpoint_after_epochs == self.save_checkpoint_after_epochs - 1 or epoch == num_epochs - 1:
                    self.__save_checkpoint(logging=True)

                if self.logger is not None:
                    self.logger.log({
                        'train_loss': train_loss,
                        'gradient_norm': train_gradient_norm,
                        'learning_rate': current_lr
                    }, self.num_steps)

            # Update learning rate
            self.scheduler.step()

            # Validation
            if val_dataloader is not None:
                self.__validate(val_dataloader, fp16=fp16)