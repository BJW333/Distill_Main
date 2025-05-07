import math, torch
import torch.nn as nn
import torch.nn.functional as F

#RMSNorm (often preferred over LayerNorm in recent LLMs)
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        # x: (batch, seq, dim)
        norm = x.norm(dim=-1, keepdim=True)
        return x / (norm / math.sqrt(x.shape[-1]) + self.eps) * self.weight

#Rotary Embeddings (RoPE): injects relative positional information via sin/cos on query/key
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        # precompute inverse frequencies for even dims
        #assert dim % 2 == 0
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
    def forward(self, seq_len, device):
        # build position indices [0..seq_len)
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        # compute outer product: (seq_len, dim/2)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # (seq, dim/2)
        # duplicate to full dim: (seq_len, dim)
        emb = torch.cat((freqs, freqs), dim=-1)            # (seq, dim)
        
        # return cosine and sine, with shape (1, seq_len, dim)
        cos, sin = emb.cos()[None, :, :], emb.sin()[None, :, :]
        return cos, sin


#helper: apply RoPE rotations to Q and K tensors
def apply_rotary(q, k, cos, sin):
    # q, k: (batch, heads, seq, dim_head)
    # split even/odd dims
    q2 = torch.stack([q[..., ::2], q[..., 1::2]], -1)  # (..., dim_head/2, 2)
    # rotate components: [x, y] -> [y, -x]
    q_rot = torch.cat((q2[...,1, :], -q2[...,0, :]), dim=-1)
    k2 = torch.stack([k[..., ::2], k[..., 1::2]], -1)
    k_rot = torch.cat((k2[...,1, :], -k2[...,0, :]), dim=-1)
    
    # combine original and rotated with cos/sin
    return (q * cos + q_rot * sin), (k * cos + k_rot * sin)


#Multi-head Attention
class SelfAttention(nn.Module):
    def __init__(self, dim, n_heads, dropout):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.d_head   = dim // n_heads
         # ← here, turn bias=False → bias=True
        self.qkv_proj = nn.Linear(dim, dim * 3, bias=True) # new lines with bias=True
        self.out_proj = nn.Linear(dim, dim,     bias=True) # new lines with bias=True
        
        #self.qkv_proj = nn.Linear(dim, dim * 3, bias=False) # old lines
        #self.out_proj = nn.Linear(dim, dim, bias=False) # old lines
        
        self.dropout  = nn.Dropout(dropout)
        self.rotary   = RotaryEmbedding(self.d_head)
        
    def forward(self, x, mask=None):
        # x: (B, S, D)
        B, S, _ = x.shape
        qkv = self.qkv_proj(x)                                  # (B, S, 3D)
        q,k,v = qkv.split(self.n_heads * self.d_head, dim=-1)   # three (B, S, D)
        # reshape to (B, heads, S, d_head)
        q = q.view(B, S, self.n_heads, self.d_head).transpose(1,2)
        k = k.view(B, S, self.n_heads, self.d_head).transpose(1,2)
        v = v.view(B, S, self.n_heads, self.d_head).transpose(1,2)
        # apply RoPE
        cos, sin = self.rotary(S, x.device)
        q, k = apply_rotary(q, k, cos, sin)
        # scaled dot-product
        attn = (q @ k.transpose(-2,-1)) * (1.0 / math.sqrt(self.d_head))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = attn @ v                                         # (B, heads, S, d_head)
        out = out.transpose(1,2).contiguous().view(B, S, -1)    # (B, S, D)
        return self.out_proj(out)

#SwiGLU Feed-Forward
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        
        # ← here, turn bias=False → bias=True
        self.w1 = nn.Linear(dim,          hidden_dim * 2, bias=True) # new lines with bias=True
        self.w2 = nn.Linear(hidden_dim,   dim,            bias=True) # new lines with bias=True
        
        #self.w1 = nn.Linear(dim, hidden_dim * 2, bias=False) #old lines
        #self.w2 = nn.Linear(hidden_dim, dim, bias=False) #old lines
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x1, x2 = self.w1(x).chunk(2, dim=-1)
        return self.dropout(self.w2(F.silu(x1) * x2))

#Transformer Block with Pre-Norm & Layer-Drop
class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, ff_dim, attn_dropout, ff_dropout, layer_drop=0.0):
        super().__init__()
        self.layer_drop = layer_drop
        self.attn_norm  = RMSNorm(dim)
        self.ff_norm    = RMSNorm(dim)
        self.attn       = SelfAttention(dim, n_heads, attn_dropout)
        self.ff         = FeedForward(dim, ff_dim, ff_dropout)
        
    def forward(self, x, mask=None):
        if self.training and torch.rand(()) < self.layer_drop:
            return x
        # attention
        a = self.attn_norm(x)
        x = x + self.attn(a, mask)
        # feed-forward
        f = self.ff_norm(x)
        x = x + self.ff(f)
        return x

#Full Decoder-only Transformer
class ModernTransformerLM(nn.Module): 
    def __init__(self,
                 vocab_size,
                 max_seq,
                 dim=768,          # Reduced dimension
                 n_heads=12,       # Adjusted heads for new dimension (768 / 12 = 64)
                 num_layers=70,      # Increased layers
                 ff_dim=3072,      # Adjusted feed-forward dimension (4 * 768)
                 dropout=0.1,
                 attn_dropout=0.1,
                 layer_drop=0.1):
        super().__init__()    
        
        self.token_emb = nn.Embedding(vocab_size, dim)
        
        #commented out below line to remove learned positional embeddings and just use RoPE
        #self.pos_emb   = nn.Parameter(torch.zeros(1, max_seq, dim)) 

        self.blocks    = nn.ModuleList([
            TransformerBlock(dim, n_heads, ff_dim, attn_dropout, dropout, layer_drop)
            for _ in range(num_layers)
        ])
        self.ln_f      = RMSNorm(dim)

        # lm_head with bias and tied weights
        self.lm_head   = nn.Linear(dim, vocab_size, bias=True)

        #could tie weights like this before applying 
        #self.lm_head.weight = self.token_emb.weight
        #then apply
        #self.apply(self._init_weights)
        
        # Initialize all weights before tying lm_head to token_emb so embeddings keep their default uniform init
        self.apply(self._init_weights)
        # Tie lm_head weights to token_emb to share parameters
        self.lm_head.weight = self.token_emb.weight
        
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, input_ids, mask=None):
        B, S = input_ids.shape

        """Modern transformer architectures that utilize RoPE (such as LLaMA, PaLM, Falcon, etc.) 
        typically remove all other forms of positional encoding, including learned or sinusoidal embeddings added to the input tokens. 
        They rely solely on RoPE applied to the Q and K vectors within the attention mechanism.
        
        Removing the learned self.pos_emb ensures that RoPE is the only method injecting positional information,
        allowing the model to leverage its benefits 
        (like better handling of sequence lengths and relative positions) without interference from a conflicting mechanism.
        """
        # The original line with learned positional embeddings below:
        # x = self.token_emb(input_ids) + self.pos_emb[:, :S, :]
        # To this:
        x = self.token_emb(input_ids) # x is now just the token embedding
        
        #(optional) activation checkpointing:
        #from torch.utils.checkpoint import checkpoint
        #for block in self.blocks:
        #    x = checkpoint(block, x, mask)

        for block in self.blocks:
            x = block(x, mask)

        x = self.ln_f(x)
        return self.lm_head(x)
    
    
