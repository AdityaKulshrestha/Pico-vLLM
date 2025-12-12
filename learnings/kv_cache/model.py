import torch
import torch.nn as nn
from dataclasses import dataclass

# LlamaForCausalLM(
#   (model): LlamaModel(
#     (embed_tokens): Embedding(49152, 576)
#     (layers): ModuleList(
#       (0-29): 30 x LlamaDecoderLayer(
#         (self_attn): LlamaAttention(
#           (q_proj): Linear(in_features=576, out_features=576, bias=False)
#           (k_proj): Linear(in_features=576, out_features=192, bias=False)
#           (v_proj): Linear(in_features=576, out_features=192, bias=False)
#           (o_proj): Linear(in_features=576, out_features=576, bias=False)
#         )
#         (mlp): LlamaMLP(
#           (gate_proj): Linear(in_features=576, out_features=1536, bias=False)
#           (up_proj): Linear(in_features=576, out_features=1536, bias=False)
#           (down_proj): Linear(in_features=1536, out_features=576, bias=False)
#           (act_fn): SiLU()
#         )
#         (input_layernorm): LlamaRMSNorm((576,), eps=1e-05)
#         (post_attention_layernorm): LlamaRMSNorm((576,), eps=1e-05)
#       )
#     )
#     (norm): LlamaRMSNorm((576,), eps=1e-05)
#     (rotary_emb): LlamaRotaryEmbedding()
#   )
#   (lm_head): Linear(in_features=576, out_features=49152, bias=False)
# )

@dataclass 
class SmolLMConfig:
    vocab_size: int = 49152
    d_model: int = 576
    num_layers: int = 2
    d_ff: int = 1536
    seq_len: int = 2048
    n_heads: int = 8
    n_kv_heads: int = 4
    eps: float = 1e-5
    rope_theta : int = 1000000  


def rotate_half(x):
    """
    Docstring for rotate_half
    
    Rotate the last dimension of the tensor by half
    1. Split the last dimension into two halves
    2. Negate the first half
    3. Swap the two halves
    4. Concatenate the two halves back together
    5. Return the rotated tensor
    
    """
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, freqs_cos, freqs_sin):
    # q, k: [batch_size, seq_len, n_heads, head_dim]
    # freqs_cos, freqs_sin: [seq_len, head_dim]
    freqs_cos = freqs_cos[None, :, None, :]  # [1, seq_len, 1, head_dim]
    freqs_sin = freqs_sin[None, :, None, :]  # [1, seq_len, 1, head_dim]
    q_embed = (q * freqs_cos) + (rotate_half(q) * freqs_sin)
    k_embed = (k * freqs_cos) + (rotate_half(k) * freqs_sin)
    return q_embed, k_embed


class SmolLMRotaryEmbedding(nn.Module):
    """
    Docstring for SmolLMRotaryEmbedding
    Good Explanation of Rotary Embeddings: https://www.wakeupsid.xyz/blog/qwen3-from-scratch
    """
    def __init__(self, config: SmolLMConfig):
        super().__init__()
        self.seq_len = config.seq_len
        self.dim = config.d_model // config.n_heads
        self.rope_theta  = config.rope_theta  

        inv_freq = 1.0 / self.rope_theta ** (torch.arange(0, self.dim, 2, dtype=torch.int64) / self.dim) # Shape: (d_model/2, )

        positions = torch.arange(self.seq_len, dtype=torch.int64)  # Shape: (seq_len,)        

        freqs = torch.outer(positions, inv_freq)  # Shape: (seq_len, d_model/2)
        freqs = torch.cat([freqs, freqs], dim=-1)  # Shape: (seq_len, d_model)

        self.register_buffer("freqs_cos", torch.cos(freqs), persistent=False)  # Shape: (seq_len, d_model/2)
        self.register_buffer("freqs_sin", torch.sin(freqs), persistent=False)  # Shape: (seq_len, d_model/2)

    def forward(self, x: torch.Tensor):
        pass


class SmolLMAttention(nn.Module):
    """
    Docstring for SmolLMAttention
    
    Reference: https://github.com/huggingface/transformers/blob/ff13eb668aa03f151ded71636d723f2e490ad967/src/transformers/models/llama/modeling_llama.py#L227
    """

    def __init__(self, config: SmolLMConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads 
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.d_model // config.n_heads

        self.q_proj = nn.Linear(config.d_model, self.head_dim * self.n_heads, bias=False)
        self.k_proj = nn.Linear(config.d_model, self.head_dim * self.n_kv_heads, bias=False)
        self.v_proj = nn.Linear(config.d_model, self.head_dim * self.n_kv_heads, bias=False)
        self.o_proj = nn.Linear(config.d_model, self.head_dim * self.n_heads , bias=False)

        # The cache storage
        self.cache_k = None
        self.cache_v = None

    def reset_cache(self):
        self.cache_k = None
        self.cache_v = None

    def forward(self, x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor, mask: torch.Tensor = None, kv_cache=None):
        bsz, seqlen, _ = x.shape


        # Compute Q, K, V for current inputs
        xq = self.q_proj(x).view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = self.k_proj(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = self.v_proj(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_pos_emb(xq, xk, freqs_cos, freqs_sin)

        # [bs, seq_len, n_heads, head_dim]      -> [bs, n_heads, seq_len, head_dim]
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # GQA -> Expand K, V to match n_heads
        if self.n_kv_heads != self.n_heads:
            xk = xk.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)
            xv = xv.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)

        scores = torch.matmul(xq, xk.transpose(2, 3)) / (self.head_dim ** 0.5)

        # Skipping the kv_cache implementation for now  

        # Apply the masking for causal attention
        if mask is None:
            mask = torch.triu(torch.full((seqlen, seqlen), float('-inf')), diagonal=1).unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
            # Replace 0s with -inf and 1s with 0
            scores = scores + mask 
        
        attn_output = torch.matmul(torch.softmax(scores, dim=-1), xv)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seqlen, self.n_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)
        return attn_output


class SmolLMRMSNorm(nn.Module):
    """
    Docstring for SmolLMRMSNorm
    
    
    """
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor):
        # RMS Norm formula:
        # RMS(a) = sqrt(mean(a^2));
        # Upscaling the precision format to avoid numerical unstability
        x = x.to(torch.float32)
        norm_x = x / torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # a = W.x; norm_a = a * (1 / RMS(a))
        return norm_x * self.weight


class SmolLMMlp(nn.Module):
    """
    Docstring for SmolLMMlp

    Reference: https://github.com/huggingface/transformers/blob/ff13eb668aa03f151ded71636d723f2e490ad967/src/transformers/models/llama/modeling_llama.py#L173
    """
    def __init__(self, config: SmolLMConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.up_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down_proj = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.act_fn = nn.SiLU()
    
    def forward(self, x: torch.Tensor):
        x = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return x


class SmolLMDecoderLayer(nn.Module):
    """
    Docstring for SmolLMDecoderLayer
    Reference: https://github.com/huggingface/transformers/blob/ff13eb668aa03f151ded71636d723f2e490ad967/src/transformers/models/llama/modeling_llama.py#L298s
    """
    def __init__(self, config: SmolLMConfig):
        super().__init__()
        self.self_attn = SmolLMAttention(config)
        self.mlp = SmolLMMlp(config)
        self.input_layernorm = SmolLMRMSNorm(config.d_model, config.eps)
        self.post_attention_layernorm = SmolLMRMSNorm(config.d_model, config.eps)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        x = self.self_attn(self.input_layernorm(x), cos, sin) + x
        x = self.mlp(self.post_attention_layernorm(x)) + x
        return x


class SmolLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([SmolLMDecoderLayer(config) for _ in range(config.num_layers)])
        self.norm = SmolLMRMSNorm(config.d_model)
        self.rotary_emb = SmolLMRotaryEmbedding(config)
        # self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # Weight tie
        # self.lm_head = self.embedding.T
        self.lm_head = self.embed_tokens


    def forward(self, x: torch.Tensor):
        x = self.embed_tokens(x)
        
        # Fetching the rotary embeddings for the sequence length
        cos = self.rotary_emb.freqs_cos[: x.shape[1], :]
        sin = self.rotary_emb.freqs_sin[: x.shape[1], :]
        for layer in self.layers:
            x = layer(x, cos, sin)
        x = self.norm(x)

        # Converting embedding to logits
        x = x @ self.lm_head.weight.T
        return x
    

    def forward_kv_cache(self, x: torch.Tensor, start_pos: int = 0, kv_cache = None):
        x = self.embed_tokens(x)
        
        # Fetching the rotary embeddings for the sequence length
        cos = self.rotary_emb.freqs_cos[start_pos: start_pos + x.shape[1], :]
        sin = self.rotary_emb.freqs_sin[start_pos: start_pos + x.shape[1], :]
        
        new_kv_cache = []
        for i, layer in enumerate(self.layers):
            layer_cache = kv_cache[i] if kv_cache is not None else None
            x = layer(x, cos, sin)
        x = self.norm(x)

        # Converting embedding to logits
        x = x @ self.lm_head.weight.T
        return x


    # generate tokens
    @torch.inference_mode()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int, use_cache: bool = False):
        if use_cache:
            kv_cache = None

            for _ in range(max_new_tokens):

                # Slicing logic for Query vector
                if kv_cache is None:
                    x_inputs = input_ids
                    start_pos = 0
                else:
                    x_inputs = input_ids[:, -1:]
                    start_pos = input_ids.shape[1] - 1
                
                logits, kv_cache = self.forward(x_inputs, start_pos=start_pos, kv_cache=kv_cache)

                next_token_logits = logits[:, -1, :]
                new_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                input_ids = torch.cat([input_ids, new_token_id])

        else: 
            for _ in range(max_new_tokens):
                outputs = self.forward(input_ids)
                new_token_id = torch.argmax(outputs[:, -1, :], dim=-1, keepdim=True)
                input_ids = torch.cat([input_ids, new_token_id], dim=-1)

        return input_ids

