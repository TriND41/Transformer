from dataclasses import dataclass

@dataclass
class TextProcessorConfig:
    tokenizer_path: str
    delim_token: str = "|"
    start_token: str = "<BOS>"
    end_token: str = "<EOS>"
    unk_token: str = "<UNK>"