import sys
sys.path.append('.')

import os
import torch
import torch.multiprocessing as mp
from tools.training import Trainer
from handlers.loading import MachineTranslationDataset
from handlers.distribution import setup, cleanup
from typing import Optional, Literal

def train(
    rank: int,
    world_size: int,
    # Training
    train_path: str,
    num_train_samples: Optional[int] = None,
    train_batch_size: int = 1,
    num_epochs: int = 1,
    fp16: bool = False,
    gradient_clipping: bool = False,
    clipping_value: Optional[float] = None,
    # Validation
    val_path: Optional[str] = None,
    num_val_samples: Optional[int] = None,
    val_batch_size: int = 1,
    # Text
    encoder_tokenizer_path: str = "./tokenizers/encoder.model",
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
    try:
        if world_size > 1:
            setup(rank, world_size)
        
        trainer = Trainer(
            rank,
            encoder_tokenizer_path, decoder_tokenizer_path, delim_token, start_token, end_token, unk_token,
            num_encoder_blocks, num_decoder_blocks, d_model, num_heads, dropout_p,
            lr, update_lr,
            checkpoint_path, checkpoint_folder, num_saved_checkpoints, save_checkpoint_after_steps, save_checkpoint_after_epochs,
            early_stopping, num_patiences, observe, score_type,
            logging, logging_project, logging_name
        )

        train_dataset = MachineTranslationDataset(
            train_path,
            trainer.get_source_tokenizer(), trainer.get_target_tokenizer(),
            num_examples=num_train_samples
        )
        val_dataset = MachineTranslationDataset(
            val_path,
            trainer.get_source_tokenizer(), trainer.get_target_tokenizer(),
            num_examples=num_val_samples
        ) if val_path is not None and os.path.exists(val_path) else None

        trainer.train(
            dataset=train_dataset,
            batch_size=train_batch_size,
            num_epochs=num_epochs,
            val_dataset=val_dataset,
            val_batch_size=val_batch_size,
            fp16=fp16,
            gradient_clipping=gradient_clipping,
            clipping_value=clipping_value
        )
    except Exception as e:
        raise ValueError(str(e))
    finally:
        if world_size > 1:
            cleanup()

def main(
    # Training
    train_path: str,
    num_train_samples: Optional[int] = None,
    train_batch_size: int = 1,
    num_epochs: int = 1,
    fp16: bool = False,
    gradient_clipping: bool = False,
    clipping_value: Optional[float] = None,
    # Validation
    val_path: Optional[str] = None,
    num_val_samples: Optional[int] = None,
    val_batch_size: int = 1,
    # Text
    encoder_tokenizer_path: str = "./tokenizers/encoder.model",
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
    assert os.path.exists(train_path)
    assert os.path.exists(encoder_tokenizer_path)

    num_gpus = torch.cuda.device_count()
    assert num_gpus > 0

    if num_gpus == 1:
        train(
            0, num_gpus,
            train_path, num_train_samples, train_batch_size,
            num_epochs, fp16, gradient_clipping, clipping_value,
            val_path, num_val_samples, val_batch_size,
            encoder_tokenizer_path, decoder_tokenizer_path, delim_token, start_token, end_token, unk_token,
            num_encoder_blocks, num_decoder_blocks, d_model, num_heads, dropout_p,
            lr, update_lr,
            checkpoint_path, checkpoint_folder, num_saved_checkpoints, save_checkpoint_after_steps, save_checkpoint_after_epochs,
            early_stopping, num_patiences, observe, score_type,
            logging, logging_project, logging_name
        )
    else:
        mp.spawn(
            fn=train,
            args=(
                num_gpus,
                train_path, num_train_samples, train_batch_size,
                num_epochs, fp16, gradient_clipping, clipping_value,
                val_path, num_val_samples, val_batch_size,
                encoder_tokenizer_path, decoder_tokenizer_path, delim_token, start_token, end_token, unk_token,
                num_encoder_blocks, num_decoder_blocks, d_model, num_heads, dropout_p,
                lr, update_lr,
                checkpoint_path, checkpoint_folder, num_saved_checkpoints, save_checkpoint_after_steps, save_checkpoint_after_epochs,
                early_stopping, num_patiences, observe, score_type,
                logging, logging_project, logging_name
            ),
            nprocs=num_gpus,
            join=True
        )

if __name__ == '__main__':
    import fire
    fire.Fire(main)