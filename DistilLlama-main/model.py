import torch
from transformers import AutoModelForCausalLM
from modern_transformer import ModernTransformerLM
from config import ModelArgs

def get_teacher_model(args: ModelArgs):
    """
    Loads and returns the pretrained LLaMA-2 (7B) teacher,
    moved to the correct device and set to eval/frozen.
    """
    teacher = AutoModelForCausalLM.from_pretrained(
        #args.teacher_model_name_or_path, 
        args.pre_trained_model_path,
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True,
        device_map="auto" # this is new and commented out teacher.to(arg.device) so model training gets distributed arcoss gpus
    )
    #teacher.to(args.device)
    teacher.eval()
    # freeze all weights
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher

def get_student_model(args: ModelArgs):
    """
    Instantiates your ModernTransformerLM with exactly the
    hyperparameters from ModelArgs, and moves it to the device.
    """
    student = ModernTransformerLM(
        vocab_size   = args.vocab_size,
        max_seq      = args.max_seq_length,
        dim          = args.dim,
        n_heads      = args.n_heads,
        num_layers   = args.n_layers,
        ff_dim       = args.dim * 4,            # or use args.ffn_dim_multiplier
        dropout      = getattr(args, "dropout", 0.1),
        attn_dropout = getattr(args, "attn_dropout", 0.1),
        layer_drop   = getattr(args, "layer_drop", 0.0),
    )
    student.to(args.device)
    return student