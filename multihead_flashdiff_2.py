import math
import torch
import torch.nn.functional as F
from torch import nn

from flash_attn import flash_attn_func

from rms_norm import RMSNorm
from einops import rearrange
from lightning_attn.ops import lightning_attn_func
from lightning_attn.utils import _build_slope_tensor
class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len=9*10000):
        super().__init__()
        # half-truncate RoPE by @YouJiacheng (w/ base freq tuning)
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim//4)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, angular_freq)
        self.register_buffer('cos', theta.cos(), persistent=False)
        self.register_buffer('sin', theta.sin(), persistent=False)
        self.max_seq_len = max_seq_len

    def forward(self, x_BTHD, offset=0):
        """
        Args:
            x_BTHD: Input tensor of shape [batch, seq_len, num_heads, dim]
            offset: Position offset for the sequence
        """
        seq_len = x_BTHD.size(-3)
        assert offset + seq_len <= self.max_seq_len, f"Sequence length {offset + seq_len} exceeds maximum length {self.max_seq_len}"

        # Get the appropriate slice of cos/sin based on offset
        cos = self.cos[None, offset:offset + seq_len, None, :]
        sin = self.sin[None, offset:offset + seq_len, None, :]

        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), dim=-1).type_as(x_BTHD)
class MultiheadFlashrope(nn.Module):
    def __init__(self, args, embed_dim, depth, num_heads):
        super().__init__()
        self.args = args
        self.embed_dim = embed_dim
        self.num_heads = num_heads  
        self.num_kv_heads = num_heads
        self.head_dim = embed_dim // num_heads 
        self.scaling = self.head_dim ** -0.5
        
        # KV cache related attributes
        self.kv_cache_enabled = False
        self.k_cache = None
        self.v_cache = None
        self.cache_pos = 0  # Track position in cache
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.rotary = Rotary(self.head_dim)
        # self.attention_scores = None
    def forward(self, x, context=None, causal=False,return_attn=False):
        """
        Args:
            x: [batch_size, tgt_len, embed_dim]
            context: Optional context for cross-attention [batch_size, src_len, embed_dim]
        """
        bsz, tgt_len, embed_dim = x.size()
        compute_attn_scores = return_attn and tgt_len == 1

        # Project inputs to q, k, v
        q = self.q_proj(x)
        k = self.k_proj(context if context is not None else x)
        v = self.v_proj(context if context is not None else x)

        # Reshape to [batch_size, seq_len, num_heads, head_dim]
        q = q.view(bsz, -1, self.num_heads, self.head_dim)
        k = k.view(bsz, -1, self.num_heads, self.head_dim)
        v = v.view(bsz, -1, self.num_heads, self.head_dim)

        # Handle KV cache if enabled
        if self.kv_cache_enabled:
            # Apply rotary embeddings with correct position offset for query
            q = self.rotary(q, offset=self.cache_pos)
            k_new = self.rotary(k, offset=self.cache_pos)
            
            # Update cache
            if self.k_cache is not None:
                cache_len = k.size(1)
                self.k_cache[:, self.cache_pos:self.cache_pos + cache_len] = k_new
                self.v_cache[:, self.cache_pos:self.cache_pos + cache_len] = v
                # Use entire cache for attention
                k = self.k_cache[:, :self.cache_pos + cache_len]
                v = self.v_cache[:, :self.cache_pos + cache_len]
                self.cache_pos += cache_len
                if compute_attn_scores:
                    # 计算当前query对所有之前位置的attention scores
                    # [batch_size, num_heads, 1, cache_len]
                    attn_weights = torch.matmul(q.transpose(1, 2), k.transpose(1, 2).transpose(-2, -1))
                    attn_weights = attn_weights * self.scaling
                    self.attention_scores = F.softmax(attn_weights, dim=-1)
                
                
        else:
            # Normal operation without cache
            q = self.rotary(q)
            k = self.rotary(k)
        # if q.dtype!=k.dtype:    
        #     print(q.dtype,k.dtype)
        # if q.dtype!=v.dtype:
        #     print(q.dtype,v.dtype)
        # Compute attention with flash attention
        # if self.training:
        #     # 训练时使用flash attention
        #     attn_output = flash_attn_func(q, k, v, causal=causal)
        # else:
        #     # 推理时使用PyTorch实现
        #     attn_output = self.attention_pytorch(q, k, v, causal=causal)
        attn_output = flash_attn_func(q, k, v, causal=causal)

        # Reshape output
        attn_output = attn_output.contiguous().view(bsz, tgt_len, embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output
    def attention_pytorch(self, q, k, v, causal=False):
        """PyTorch标准注意力实现，用于推理"""
        # [batch_size, num_heads, seq_len, head_dim]
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # 计算注意力分数
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling

        if causal:
            # 创建因果mask
            seq_len = q.size(-2)
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=q.device),
                diagonal=1
            )
            attn_weights.masked_fill_(causal_mask, float('-inf'))

        # 应用softmax
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # 计算输出
        attn_output = torch.matmul(attn_weights, v)
        
        # 恢复形状 [batch_size, seq_len, num_heads, head_dim]
        return attn_output.permute(0, 2, 1, 3)

    def empty_kv_cache(self, batch_size: int, kv_cache_maxlen: int, dtype: torch.dtype):
        """Initialize empty KV cache"""
        self.kv_cache_enabled=True

        device = next(self.parameters()).device
        
        # Initialize empty caches
        k_cache = torch.zeros(
            batch_size, 
            kv_cache_maxlen, 
            self.num_heads,
            self.head_dim,
            dtype=dtype,
            device=device
        )
        self.k_cache=k_cache

        v_cache = torch.zeros(
            batch_size,
            kv_cache_maxlen,
            self.num_heads,
            self.head_dim,
            dtype=dtype,
            device=device
        )       
        self.v_cache=v_cache
        self.cache_pos = 0  # Reset cache position

    def reset_cache(self):
        """Reset KV cache state"""
        self.k_cache = None
        self.v_cache = None
        self.cache_pos = 0
class MultiheadFlashlinearrope(nn.Module):
    def __init__(self, args, embed_dim, depth, num_heads):
        super().__init__()
        self.args = args
        self.embed_dim = embed_dim
        self.num_heads = num_heads  
        self.num_kv_heads = num_heads
        self.head_dim = embed_dim // num_heads 
        self.scaling = self.head_dim ** -0.5
        
        # KV cache related attributes
        self.kv_cache_enabled = False
        self.kv_cache = None
        self.cache_pos = 0  # Track position in cache
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.rotary = Rotary(self.head_dim)
        self.rms_norm = RMSNorm(embed_dim, eps=1e-5)
        # self.output_gate = nn.Linear(embed_dim, embed_dim, bias=False)
        # self.output_gate = nn.Sequential(
        #     nn.Linear(embed_dim, self.head_dim, bias=False),
        #     nn.Linear(self.head_dim, embed_dim, bias=False),
        # )
        
        slope_rate = _build_slope_tensor(self.num_heads)
        # slope_rate = slope_rate * (1 - depth / (24 - 1) + 1e-5)
        self.register_buffer("slope_rate", slope_rate, persistent=False)

    def forward(self, x, context=None, causal=False):
        bsz, seq_len, _ = x.size()
        

        q = self.q_proj(x)
        k = self.k_proj(context if context is not None else x)
        v = self.v_proj(context if context is not None else x)

        # Reshape: [batch_size, seq_len, num_heads, head_dim]
        # Reshape for multi-head attention
        q = q.view(bsz, -1, self.num_heads, self.head_dim)
        k = k.view(bsz, -1, self.num_heads, self.head_dim)
        v = v.view(bsz, -1, self.num_heads, self.head_dim)

        # q = F.silu(q)
        # k = F.silu(k)
        # v = F.silu(v)

        # s = _build_slope_tensor( self.num_heads).to(q.device).to(torch.float32)



        if self.kv_cache_enabled: 
            ratio = torch.exp(-self.slope_rate).view(1, self.num_heads, 1, 1) 
            q, k = self.rotary(q, offset=self.cache_pos), self.rotary(k, offset=self.cache_pos)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            
            # decay_rate = torch.exp(self.slope_rate).view(1, self.num_heads, 1, 1)  # [num_heads, 1, 1]
            
            if self.kv_cache is None:
                kv_cache = torch.zeros(bsz, self.num_heads, self.head_dim, self.head_dim, 
                                          device=q.device, dtype=q.dtype)
                self.kv_cache=kv_cache


            output = []
            for i in range(seq_len):
                self.kv_cache = ratio * self.kv_cache + torch.einsum(
                    "... n d, ... n e -> ... d e",
                    k[:, :, i:i + 1],
                    v[:, :, i:i + 1],
                )
                qkv = torch.einsum("... n e, ... e d -> ... n d", q[:, :, i:i + 1], self.kv_cache)
                output.append(qkv)
            attn_output = torch.concat(output, dim=-2)
            
            self.cache_pos += seq_len
            
        else:
  
            q, k = self.rotary(q), self.rotary(k)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)   
            attn_output = lightning_attn_func(q, k, v,self.slope_rate)


        attn_output = rearrange(attn_output, 'b h n d -> b n (h d)')
        attn_output = self.rms_norm(attn_output)

        attn_output = self.out_proj(attn_output)

        return attn_output


    def empty_kv_cache(self, batch_size: int, kv_cache_maxlen: int, dtype: torch.dtype):
        """Initialize empty KV cache"""
        self.kv_cache_enabled=True
        self.cache_pos = 0  # Reset cache position

    def reset_cache(self):
        """Reset KV cache state"""
        self.kv_cache = None
        self.cache_pos = 0


