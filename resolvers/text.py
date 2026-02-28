import os
import numpy as np
import re
import multiprocessing as mp
from handlers.common import load_configs, compute_parallel_processes
from abc import ABC, abstractmethod
import sentencepiece as spm
from typing import List, Tuple, Optional, Dict

# Base Text Processor
class TextProcessor(ABC):
    def __init__(
        self,
        tokenizer_path: str,
        delim_token: str = "|",
        start_token: str = "<BOS>",
        end_token: str = "<EOS>",
        unk_token: str = "<UNK>"
    ) -> None:
        self.tokenizer_data = load_configs(tokenizer_path) if ".json" in tokenizer_path else None

        self.start_token = start_token
        self.delim_token = delim_token
        self.end_token = end_token
        self.unk_token = unk_token

    @staticmethod
    def process_sequences(sequences: List[np.ndarray], pad_value: int) -> Tuple[np.ndarray, np.ndarray]:
        lengths = np.array([len(sequence) for sequence in sequences], dtype=np.int32)
        max_length = lengths.max()

        padded_audios = []
        for sequence, length in zip(sequences, lengths):
            padded_audios.append(
                np.pad(
                    sequence,
                    pad_width=[0, max_length - length],
                    mode='constant',
                    constant_values=pad_value
                )
            )
        
        return np.stack(padded_audios, axis=0), lengths
    
    @staticmethod
    def find_token_id(vocabs: List[str], token: str) -> int:
        return vocabs.index(token)
    
    @staticmethod
    def find_token_by_id(vocabs: List[str], token_id: int) -> str:
        return vocabs[token_id]
    
    @staticmethod
    def map_sequence(text: str, mapping_dict: Dict[str, str]) -> str:
        for pattern, replacement in mapping_dict.items():
            text = text.replace(pattern, replacement)
        return text
    
    @abstractmethod
    def encode(self, text: str) -> np.ndarray:
        pass

    @abstractmethod
    def decode(self, sequence: np.ndarray, text_lowercase: bool = False, text_uppercase: bool = False, mapping_dict: Optional[Dict[str, str]] = None) -> str:
        pass

    @abstractmethod
    def convert(self, sequence: np.ndarray) -> str:
        pass

    def batch_decode(
        self,
        sequences: np.ndarray,
        text_lowercase: bool = False,
        text_uppercase: bool = False,
        mapping_dict: Optional[Dict[str, str]] = None,
        parallel_decoding: bool = False,
        num_parallel_decoding_processes: Optional[int] = None
    ) -> List[str]:
        if not parallel_decoding:
            texts = [self.decode(sequence, text_lowercase, text_uppercase, mapping_dict) for sequence in sequences]
        else:
            # Find optimal Number of Processes for parallel
            num_parallel_decoding_processes = compute_parallel_processes(len(sequences), num_parallel_decoding_processes)
            
            # Run Parallel
            input_args = [(sequence, text_lowercase, text_uppercase, mapping_dict) for sequence in sequences]
            with mp.Pool(num_parallel_decoding_processes) as pool:
                texts = pool.starmap(self.decode, input_args)
            
        return texts
    
    @staticmethod
    def autoregressive_pad(tokens: np.ndarray, start_id: int, end_id: int) -> np.ndarray:
        tokens = np.pad(tokens, [(0, 0), (1, 0)], mode='constant', constant_values=start_id)
        tokens = np.pad(tokens, [(0, 0), (0, 1)], mode='constant', constant_values=end_id)
        return tokens
        
    @abstractmethod
    def __call__(self, sequences: List[np.ndarray], auto_regressive: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        pass

class TextGraphemeProcessor(TextProcessor):
    def __init__(
        self,
        tokenizer_path: str,
        delim_token: str = "|",
        start_token: str = "<BOS>",
        end_token: str = "<EOS>",
        unk_token: str = "<UNK>"
    ) -> None:
        super().__init__(tokenizer_path, delim_token, start_token, end_token, unk_token)

        self.vocabs = list(self.tokenizer_data['vocabs'])
        self.num_vocabs = len(self.vocabs)
        self.replace_dict = dict(self.tokenizer_data['replace_dict'])

        self.start_id = self.find_token_id(self.vocabs, start_token)
        self.end_id = self.find_token_id(self.vocabs, end_token)
        self.delim_id = self.find_token_id(self.vocabs, delim_token)
        self.unk_id = self.find_token_id(self.vocabs, unk_token)

    def __spec_decode(self, text: str) -> str:
        for pattern, replacement in self.replace_dict.items():
            text = re.sub(pattern, replacement, text)
        return text
    
    def encode(self, text: str) -> np.ndarray:
        subwords = text.split(" ")
        ids = []
        for subword in subwords:
            ids.append(self.find_token_id(self.vocabs, subword))
        return np.array(ids, dtype=np.int32)
    
    def decode(
        self,
        sequence: np.ndarray,
        text_lowercase: bool = False,
        text_uppercase: bool = False,
        mapping_dict: Optional[Dict[str, str]] = None
    ) -> str:
        # Convert IDs to Textual Items
        subwords = []
        for token_id in sequence:
            if token_id == self.end_id:
                break
            elif token_id != self.delim_id:
                subwords.append(self.find_token_by_id(self.vocabs, token_id))
            else:
                subwords.append(" ")

        # Concatenate found sub-words
        text = self.__spec_decode("".join(subwords))

        # Text Handling
        if text_lowercase:
            text = str(text).lower()
        if text_uppercase:
            text = str(text).upper()

        # Mapping (Post-Processing)
        if mapping_dict is not None:
            text = self.map_sequence(text, mapping_dict)

        return text
    
    def __call__(self, sequences: List[np.ndarray], auto_regressive: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        # Process the different length of audios
        sequences, lengths = self.process_sequences(sequences)
        """
            sequences: Batch of padded textual sequences, shape = [num_sequences, length], length is the max of sequence lengths
            lengths: The length of each sequence, shape = [num_sequences]
        """

        # Setup Input Representation (Optional for Auto-Regressive Modeling)
        if auto_regressive:
            sequences = self.autoregressive_pad(sequences, self.start_id, self.end_id)
            lengths += 2

        return sequences, lengths
    
class TextBPEProcessor(TextProcessor):
    def __init__(
        self,
        tokenizer_path: str,
        delim_token: str = "|",
        start_token: str = "<BOS>",
        end_token: str = "<EOS>",
        unk_token: str = "<UNK>"
    ) -> None:
        super().__init__(tokenizer_path, delim_token, start_token, end_token, unk_token)

class TextSentencePieceProcessor(TextProcessor):
    def __init__(
        self,
        tokenizer_path: str,
        delim_token: str = "|",
        start_token: str = "<BOS>",
        end_token: str = "<EOS>",
        unk_token: str = "<UNK>",
        num_extra: int = 96
    ) -> None:
        super().__init__(tokenizer_path, delim_token, start_token, end_token, unk_token)
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.Load(tokenizer_path)
        self._num_extra = num_extra

        self.__num_tokens = self.tokenizer.vocab_size()
        self.num_vocabs = self.__num_tokens + num_extra
        
        # Determine ID of Special Tokens
        current_count = 1

        self.start_id = self.tokenizer.bos_id()
        if self.start_id < 0:
            self.start_id = self.__num_tokens + current_count
            current_count += 1
        
        self.end_id = self.tokenizer.eos_id()
        if self.end_id < 0:
            self.start_id = self.__num_tokens + current_count
            current_count += 1

        self.unk_id = self.tokenizer.unk_id()
        if self.unk_id < 0:
            self.unk_id = self.__num_tokens + current_count
            current_count += 1

    def encode(self, text: str) -> np.ndarray:
        ids = np.array(
            self.tokenizer.Encode(text, add_bos=False, add_eos=False, out_type=int),
            dtype=np.int32
        )
        return ids
    
    def decode(
        self,
        sequence: np.ndarray,
        text_lowercase: bool = False,
        text_uppercase: bool = False,
        mapping_dict: Optional[Dict[str, str]] = None
    ) -> str:
        text = self.tokenizer.Decode(sequence.tolist())
        if mapping_dict is not None:
            text = self.map_sequence(text, mapping_dict)
        return text
    
    def __call__(self, sequences: List[str], auto_regressive: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        tokenized_sequences = [self.encode(sequence) for sequence in sequences]
        tokens, token_lengths = self.process_sequences(tokenized_sequences, pad_value=self.end_id)
        if auto_regressive:
            tokens = self.autoregressive_pad(tokens, self.start_id, self.end_id)
            token_lengths += 2
        return tokens, token_lengths
    
class TextCharacterBasedProcessor(TextProcessor):
    def __init__(
        self,
        tokenizer_path: str,
        delim_token: str = "|",
        start_token: str = "<BOS>",
        end_token: str = "<EOS>",
        unk_token: str = "<UNK>"
    ) -> None:
        super().__init__(tokenizer_path, delim_token, start_token, end_token, unk_token)
        self.vocabs = list(self.tokenizer_data)
        self.num_vocabs = len(self.vocabs)

        self.start_id = self.find_token_id(self.vocabs, start_token)
        self.end_id = self.find_token_id(self.vocabs, end_token)
        self.delim_id = self.find_token_id(self.vocabs, delim_token)
        self.unk_id = self.find_token_id(self.vocabs, unk_token)

    def encode(self, text: str) -> np.ndarray:
        subwords = [*text]
        ids = []
        for subword in subwords:
            ids.append(self.find_token_id(self.vocabs, subword))
        return np.array(subwords, dtype=np.int32)
    
    def decode(
        self,
        sequence: np.ndarray,
        text_lowercase: bool = False,
        text_uppercase: bool = False,
        mapping_dict: Optional[Dict[str, str]] = None
    ) -> str:
        subwords = []
        for token_id in sequence:
            if token_id == self.end_id:
                break
            elif token_id != self.delim_id:
                subwords.append(self.find_token_by_id(self.vocabs, token_id))
            else:
                subwords.append(" ")

        # Concatenate found sub-words
        text = "".join(subwords)

        # Text Handling
        if text_lowercase:
            text = str(text).lower()
        if text_uppercase:
            text = str(text).upper()

        # Mapping (Post-Processing)
        if mapping_dict is not None:
            text = self.map_sequence(text, mapping_dict)

        return text
    
    def __call__(self, sequences: List[np.ndarray], auto_regressive: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        # Process the different length of audios
        sequences, lengths = self.process_sequences(sequences)
        """
            sequences: Batch of padded textual sequences, shape = [num_sequences, length], length is the max of sequence lengths
            lengths: The length of each sequence, shape = [num_sequences]
        """

        # Setup Input Representation (Optional for Auto-Regressive Modeling)
        if auto_regressive:
            sequences = self.autoregressive_pad(sequences, self.start_id, self.end_id)
            lengths += 2

        return sequences, lengths