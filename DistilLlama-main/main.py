import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
#import sentencepiece as spm
from pathlib import Path
from torch.utils.data import ConcatDataset

from arguments import get_args
from config import ModelArgs, TrainArgs
from data import TextFileDataset, llama_collate_fn #ParquetDataset,
from modern_transformer import ModernTransformerLM
from transformers import LlamaForCausalLM, LlamaTokenizer

# Make the old import still work:
Transformer = ModernTransformerLM
from train import train

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

# # Local test
# def setup(rank, world_size): # 
#     os.environ['MASTER_ADDR'] = 'localhost' # 
#     os.environ['MASTER_PORT'] = '12355' # 
#     dist.init_process_group("nccl", rank=rank, world_size=world_size) # 


def cleanup():
    dist.destroy_process_group()

def run_main():
    print("▶︎ Starting run_main()")    # <<<<< add this
    args = get_args()
    print("▶︎ Full args:", args)
    #print(f"▶︎ Using device: cuda:{local_rank}")
    print(f"▶︎ batch_size={args['batch_size']}, accum_steps={args['accumulation_steps']}")
    #tokenizer = spm.SentencePieceProcessor()
    #tokenizer.load(args['tokenizer_model_path'])
    #tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b")
    tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

    
    #wiki_dataset = ParquetDataset(args['wikitext_path'], tokenizer, args['max_seq_length'], pad_token_id=0)
    wiki_dataset = TextFileDataset(args['wikitext_path'], tokenizer, args['max_seq_length'], pad_token_id=0)
    
    openweb_dataset = TextFileDataset(args['openwebtext_path'], tokenizer, args['max_seq_length'], pad_token_id=0)
    
    dataset = ConcatDataset([wiki_dataset, openweb_dataset])
    
    dataset_size = len(dataset)
    train_size = int(dataset_size * args['train_ratio'])
    eval_size = dataset_size - train_size

    train_dataset, eval_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, eval_size],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"▶︎ train examples: {len(train_dataset)}  |  eval examples: {len(eval_dataset)}")

    
    if "SLURM_PROCID" in os.environ:
        world_size = int(os.environ["WORLD_SIZE"])
        rank       = int(os.environ["SLURM_PROCID"])
        gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
    else:
        world_size = int(os.environ["WORLD_SIZE"])
        rank       = int(os.environ["RANK"])
        gpus_per_node = torch.cuda.device_count()
    local_rank = rank - gpus_per_node * (rank // gpus_per_node)
    assert gpus_per_node == torch.cuda.device_count()
    
    setup(rank, world_size)
    
    print(f"Hello from rank {rank} of {world_size} where there are {gpus_per_node} allocated GPUs per node.", flush=True)

    
    torch.cuda.set_device(local_rank)
    print(f"▶︎ Using device: cuda:{local_rank}")
    print(f"▶︎ Using {gpus_per_node} GPUs per node")
    print(f"▶︎ Using {world_size} GPUs total")
    print(f"▶︎ Using {rank} as rank")
    print(f"▶︎ Using {local_rank} as local rank")
    print(f"▶︎ Using {dataset_size} total examples")
    print(f"▶︎ Using {train_size} train examples")
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=42)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args['batch_size'],
        sampler=train_sampler,
        num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]), # 
        #num_workers=4, # For testing
        pin_memory=True,
        collate_fn=llama_collate_fn
    )

    eval_sampler = DistributedSampler(eval_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args['batch_size'],
        sampler=eval_sampler,
        num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]), # 
        #num_workers=4, # For testing
        pin_memory=True,
        collate_fn=llama_collate_fn
    )

    # ─── build student ModelArgs ────────────────────────────────────────
    student_model_args = ModelArgs()
    
    # 1) vocab from your tokenizer
    #student_model_args.vocab_size = tokenizer.vocab_size()
    student_model_args.vocab_size = tokenizer.vocab_size

    # 2) your RMSNorm+RoPE model settings
    student_model_args.dim            = 768
    student_model_args.n_layers       = 70
    student_model_args.n_heads        = 12
    
    # (leave n_kv_heads=None unless you want fewer KV heads)
    # 3) sequence length for RoPE and positional cache
    student_model_args.max_seq_length = args['max_seq_length']
    
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    # build student
    student = Transformer(device=device, args=student_model_args)
    student = student.to(device)
    student_model = DDP(student, device_ids=[local_rank])
    # now build & wrap your student
    
    # teacher_model_args = ModelArgs(
    #     dim=4096,
    #     n_layers=32,
    #     n_heads=32,
    #     vocab_size=student_model_args.vocab_size,
    #     multiple_of=256,
    #     norm_eps=1e-5,
    #     max_seq_length=args['max_seq_length'],
    #     pre_trained_model_path='llama-2-7b'
    # )
    # build teacher
    #teacher = Transformer(device=device, args=teacher_model_args)
    #teacher_model = teacher.to(device=device, dtype=torch.bfloat16)
    
    teacher_model = LlamaForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-chat-hf", #meta-llama/Llama-2-7b
        torch_dtype=torch.bfloat16,      # or float16 if you prefer
        #torch_dtype=torch.float16,  #mac handles float16 better
        low_cpu_mem_usage=True,
        device_map="auto"                # will shard across GPUs if needed
    )
    #teacher_model = teacher_model.to(device)
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad = False
        
    # Load teacher model weights
    # checkpoints = sorted(Path(teacher_model_args.pre_trained_model_path).glob('*.pth'))
    # assert len(checkpoints) > 0
    # chk_path = checkpoints[0]
    # checkpoint = torch.load(chk_path, map_location='cpu')  # Load to CPU first
    # if 'rope.freqs' in checkpoint:
    #     del checkpoint['rope.freqs']
    # teacher_model.load_state_dict(checkpoint)
    # del checkpoint  # Free up memory
    # torch.cuda.empty_cache()

    train_config = TrainArgs()
    
    print("▶︎ Calling train() now…")
    train(
        rank=rank,
        device=device, # not device=local_rank 
        student_model=student_model,
        teacher_model=teacher_model,
        train_config=train_config,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        args=args
    )
    print("▶︎ train() returned")
    
    try:
        cleanup()
    except Exception:
        pass

if __name__ == '__main__':
    run_main()
