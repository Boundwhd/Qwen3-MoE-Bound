import torch
import torch.nn as nn
from typing import Optional, Tuple

'''
  Qwen3Config
'''
class Qwen3Config:
    def __init__(
        self,
        vocab_size=151936,  
        hidden_size=32,
        intermediate_size=3072,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=4,
        max_position_embeddings=5,
        rms_norm_eps=1e-6,
        rope_theta=1000000,
        use_cache=True
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta=rope_theta
        self.use_cache=use_cache

'''
  RMSNorm
'''
class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x:torch.Tensor):
        norm_x = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm_x + self.eps)
        return x
    
'''
  Rope Position Embedding
'''
def _rotate_half(x) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def _apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)  # [b, 1, seq_len, dim]
    sin = sin.unsqueeze(unsqueeze_dim)  # [b, 1, seq_len, dim]
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed

def _inv_freq_compute(config:Qwen3Config, device) -> torch.Tensor:
    base = config.rope_theta
    head_dim = config.hidden_size // config.num_attention_heads
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float32) / head_dim))
    return inv_freq

class RopePositionEmbedding(nn.Module):
    def __init__(self, config:Qwen3Config, device):
        super().__init__()
        self.device = device
        self.max_seq_len = config.max_position_embeddings
        inv_freq = _inv_freq_compute(config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq
    
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device) # shape [batch, head_dim // 2, 1]
        position_ids_expanded = position_ids[:, None, :].float() # shape [batch, 1, seq_len]
        with torch.autocast(device_type=self.device, enabled=False): # Float32 calculate
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2) # [batch, seq_len, head_dim//2]
            emb = torch.cat((freqs, freqs), dim=-1) # [batch, seq_len, head_dim]
            cos = emb.cos() 
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

'''
  MLP
'''
class MLP(nn.Module):
    def __init__(self, config:Qwen3Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()
    
    def forward(self, x:torch.Tensor):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

'''
 Attention with GQA
'''
class kv_cache:
    def __init__(self, config:Qwen3Config, device):
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.max_seq_len = config.max_position_embeddings
        self.key_cache = torch.zeros((1, self.max_seq_len, config.num_key_value_heads, self.head_dim), device=device)
        self.value_cache = torch.zeros((1, self.max_seq_len, config.num_key_value_heads, self.head_dim), device=device)
    
    def decode_update(self, k, v, cache_position):
        if cache_position > self.max_seq_len:
            raise ValueError("Current token exceeds maximum sequence length")
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        self.key_cache[:,cache_position, :] = k
        self.value_cache[:,cache_position, :] = v

    def prefill_update(self, k, v, prompt_seq_len):
        if prompt_seq_len > self.max_seq_len:
            raise ValueError("Prompt exceeds maximum sequence length")
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        self.key_cache[:, 0:prompt_seq_len] = k
        self.value_cache[:, 0:prompt_seq_len] = v

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class Attention(nn.Module):
    def __init__(self, config:Qwen3Config):
        super().__init__()
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.num_group = config.num_attention_heads // config.num_key_value_heads

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        prefill_stage: bool,
        past_key_value: kv_cache,
        cache_position: int
    ) -> torch.Tensor:
        input_shape = hidden_states.shape[:-1] # [batch, seq_len]
        hidden_shape = (*input_shape, -1, self.head_dim) # [batch, seq_len, num_attention_heads, head_dim]

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2) # [b, h, s, d]
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = _apply_rotary_pos_emb(query_states, key_states, cos, sin) 

        if prefill_stage:
            past_key_value.prefill_update(key_states, value_states, hidden_states.shape[1])
        else :
            past_key_value.decode_update(key_states, value_states, cache_position)
            key_cache = past_key_value.key_cache
            value_cache = past_key_value.value_cache
            key_states = key_cache[:,:cache_position + 1].transpose(1, 2)
            value_states = value_cache[:,:cache_position + 1].transpose(1, 2)

        key_states = repeat_kv(key_states, self.num_group)
        value_states = repeat_kv(value_states, self.num_group)

        attention_scores = query_states @ key_states.transpose(-1, -2) / self.scaling
        attention_probs = torch.softmax(attention_scores, dim=-1)
        attention_output = attention_probs @ value_states
        attention_output = self.o_proj((attention_output.transpose(1, 2).contiguous()).view(*input_shape, -1))
        return attention_output

class DecoderLayer(nn.Module):
    def __init__(self, config:Qwen3Config, device):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.kv_cache = kv_cache(config, device=device)
        self.self_attn = Attention(config=config)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        prefill_stage: bool,
        position_embedding: Tuple[torch.Tensor, torch.Tensor],
        cache_position: int
    ) -> torch.Tensor :
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            position_embedding,
            prefill_stage,
            self.kv_cache,
            cache_position
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        output = residual + hidden_states

        return output
    
class Qwen3Model(nn.Module):
    def __init__(self, config:Qwen3Config, device):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [DecoderLayer(config, device) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = RopePositionEmbedding(config, device)
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        prefill_stage: bool,
        cache_position: int,
    ) -> torch.tensor:
        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds

        position_ids = torch.arange(0, inputs_embeds.shape[1], dtype=torch.int) if prefill_stage \
            else torch.tensor([cache_position], dtype=torch.int)
        
        position_ids = position_ids.unsqueeze(0)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        

        for decoder_layer in self.layers:
            layer_output = decoder_layer(
                hidden_states,
                prefill_stage,
                position_embeddings,
                cache_position
            )
            hidden_states = layer_output

        hidden_states = self.norm(hidden_states)
        return hidden_states
        

class Qwen3ForCausalLM(nn.Module):
    def __init__(self, config:Qwen3Config, device):
        super().__init__()
        self.model = Qwen3Model(config, device)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self, 
        input_ids: torch.LongTensor,
        prefill_stage: bool,
        cache_position: int = None,
    ):
        hidden_states = self.model(input_ids, prefill_stage, cache_position)
        output_states = self.lm_head(hidden_states)
        next_ids = torch.argmax(output_states[0, -1], dim=-1)
        next_ids = torch.tensor([next_ids]).unsqueeze(0)
        return next_ids


if __name__ == "__main__":

    input_ids = torch.tensor([[1, 4, 5]], dtype=torch.int)
    output_ids = torch.tensor([],dtype=torch.int)
    cache_pos = None
    model = Qwen3ForCausalLM(config=Qwen3Config(), device="cpu")
    next_ids = model(input_ids, True)
    out_token = next_ids.squeeze(dim=0)
    output_ids = torch.cat((output_ids, out_token), dim=-1)
    if cache_pos == None:
        cache_pos = input_ids.shape[1]
    else:
        cache_pos = cache_pos + 1
    next_ids = model(next_ids, False, cache_pos)
    out_token = next_ids.squeeze(dim=0)
    output_ids = torch.cat((output_ids, out_token), dim=-1)
    print(output_ids)







