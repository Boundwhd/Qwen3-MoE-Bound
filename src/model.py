import torch
import torch.nn as nn

class QwenConfig:
    def __init__(
        self,
        vocab_size=151936,  
        hidden_size=1024,
        intermediate_size=3072,
        num_hidden_layers=28,
        num_attention_heads=16,
        num_key_value_heads=8,
        max_position_embeddings=2048,
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
    def __init__(self, config:QwenConfig):
        super().__init__()
        self.eps = config.rms_norm_eps
        self.weight = nn.Parameter(torch.ones(config.hidden_size))

    def forward(self, x: torch.tensor):
        norm_x = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm_x + self.eps)
        return x
    
'''
  Rope Position Embedding
'''
def _inv_freq_compute(config:QwenConfig, device) -> torch.tensor:
    base = config.rope_theta
    head_dim = config.hidden_size // config.num_attention_heads
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / head_dim))
    return inv_freq

class RopeEmbedding(nn.Module):
    def __init__(self, config:QwenConfig, device):
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
    





if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = QwenConfig()
    RMSNorm_layer = RMSNorm(config)
    RopeEmbedding_layer = RopeEmbedding(config, device)

    data = torch.randn([1, 10, 64], dtype=torch.float32)
    print(data.device)
    position = torch.arange(10).expand(1, -1)
    cos, sin = RopeEmbedding_layer(data, position)
    print(sin[0,7,:10])



