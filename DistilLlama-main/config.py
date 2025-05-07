from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelArgs:
    dim: int = 768
    n_layers: int = 70
    n_heads: int = 12
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    mode: str = 'train'
    batch_size: int = 64 # this is used when more computation space is avaiable 
    #batch_size: int = 8    # <-- change this if you run out of memory/computation space
    max_seq_length: int = 1024
    #max_seq_length: int = 512 smaller max seq length cause u dont need as long context things/conversations if ur computing cant handle in training
    dropout: float = 0.1
    attn_dropout: float = 0.1
    layer_drop: float = 0.1
    # teacher & device
    pre_trained_model_path: Optional[str] = "meta-llama/Llama-2-7b-chat-hf"
    pre_trained_tokenizer_path: Optional[str] = "tokenizer.model"
    device: str = "auto"
    #pre_trained_model_path: Optional[str] = None
    #pre_trained_tokenizer_path: Optional[str] = None

@dataclass
class TrainArgs(ModelArgs):
    n_epochs: int = 3
    log_interval: int = 3000
    lr: float = 3e-4
    warmup_steps: int = 4000
    accumulation_steps: int = 2 # this results in bigger 
    #accumulation_steps: int = 16   # <-- increases gradient accumulation so micro-batches are smaller
    load_model: bool = True
    temperature: float = 2.0
    alpha: float = 0.3
    n_random_sample: int = 5000
    save_dir: str = 'DistilLlama-Checkpoints'

# # Local test
# @dataclass
# class ModelArgs:
#     dim: int = 512
#     n_layers: int = 16
#     n_heads: int = 8
#     n_kv_heads: Optional[int] = None
#     vocab_size: int = -1
#     multiple_of: int = 256
#     ffn_dim_multiplier: Optional[float] = None
#     norm_eps: float = 1e-5
#     mode: str = 'train'
#     batch_size: int = 2
#     max_seq_length: int = 32
#     pre_trained_model_path: Optional[str] = None
#     pre_trained_tokenizer_path: Optional[str] = None

# # Local test
# @dataclass
# class TrainArgs(ModelArgs):
#     n_epochs: int = 2
#     log_interval: int = 10
#     lr: float = 2.5e-4
#     warmup_steps: int = 100
#     accumulation_steps: int = 16
#     load_model: bool = True
#     temperature: float = 2
#     alpha: float = 0.5
#     n_random_sample: int = 100
#     save_dir: str = 'DistilLlama-Checkpoints'

@dataclass
class DataArgs(ModelArgs):
    #wikitext_path: str = 'wikitext/wikitext-103-raw-v1'
    wikitext_path: str = 'wikitext/wikitext-103-raw-v1/wiki.train.raw'
    #openwebtext_path: str = 'openwebtext_dataset1.txt' # This is one half of the dataset
    openwebtext_path: str = 'openwebtext/subsets/openwebtext.txt'
    tokenizer_model_path: str = 'tokenizer.model'
    train_ratio: float = 0.9
    
@dataclass
class InferenceArgs(ModelArgs):
    checkpoint_dir: str = 'DistilLlama-Checkpoints'
    tokenizer_path: str = 'tokenizer.model'
    load_model: bool = True
    max_seq_len: int = 64
    temperature: float = 0.7
    top_p: float = 0.4
