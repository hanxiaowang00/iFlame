from multihead_flashdiff_2 import MultiheadFlashrope,MultiheadFlashlinearrope
from rms_norm import RMSNorm
import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Tuple
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import tqdm as tqdm1
import timm


from torch.optim.lr_scheduler import LambdaLR    
import subprocess
import math
from einops import rearrange, repeat, pack


class SwiGLU(nn.Module):
    def __init__(self, embed_dim):
        super(SwiGLU, self).__init__()
        self.proj1 = nn.Linear(embed_dim, int(8 / 3 * embed_dim))  # XWG
        self.proj2 = nn.Linear(embed_dim, int(8 / 3 * embed_dim))  # XW1
        self.proj3 = nn.Linear(int(8 / 3 * embed_dim), embed_dim)  # W2

    def forward(self, x):
        # Apply Swish (SiLU in PyTorch) to the first projection
        x_proj1 = F.silu(self.proj1(x))  # Swish(XWG)
        
        # Apply the second projection
        x_proj2 = self.proj2(x)  # XW1
        
        # Element-wise multiplication
        x_glu = x_proj1 * x_proj2  # (swish(XWG) ⊙ XW1)
        
        # Final projection back to the original dimension
        output = self.proj3(x_glu)  # (swish(XWG) ⊙ XW1)W2
        
        return output
class PointEmbed(nn.Module):
    def __init__(self, hidden_dim=48, dim=128):
        super().__init__()

        assert hidden_dim % 6 == 0

        self.embedding_dim = hidden_dim
        e = torch.pow(2, torch.arange(self.embedding_dim // 6)).float() * np.pi
        e = torch.stack([
            torch.cat([e, torch.zeros(self.embedding_dim // 6),
                        torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6), e,
                        torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6),
                        torch.zeros(self.embedding_dim // 6), e]),
        ])
        self.register_buffer('basis', e)  # 3 x 16

        self.mlp = nn.Linear(self.embedding_dim+3, dim)

    @staticmethod
    def embed(input, basis):
        projections = torch.einsum(
            'bnd,de->bne', input, basis)
        embeddings = torch.cat([projections.sin(), projections.cos()], dim=2)
        return embeddings
    
    def forward(self, input):
        # input: B x N x 3
        embed = self.mlp(torch.cat([self.embed(input, self.basis), input], dim=2)) # B x N x C
        return embed

def shift_sequence(x, shift_amount):

    if shift_amount == 0:
        return x
    else:
        shifted_x = torch.zeros_like(x)
        if shift_amount < x.size(1):
            shifted_x[:, shift_amount:, :] = x[:, :-shift_amount, :]

        return shifted_x


def pad_to_multiple(tensor, multiple, dim=-1, pad_value=0):

    current_size = tensor.size(dim)
    

    remainder = current_size % multiple
    if remainder == 0:
        return tensor  
    
    pad_size = multiple - remainder


    pad = [0] * (2 * tensor.dim())

    pad_start = (tensor.dim() - 1 - dim) * 2
    pad[pad_start + 1] = pad_size 


    padded_tensor = F.pad(tensor, pad, mode='constant', value=pad_value)
    return padded_tensor

class DifferentialTransformerBlockrope(nn.Module):
    def __init__(self, embed_dim, num_heads, depth, args, causal=False):
        """
        Differential Transformer Block with optional causal self-attention or cross-attention
        and automatic application of ROPE.
        
        Args:
            embed_dim (int): Embedding dimension.
            num_heads (int): Number of attention heads.
            depth (int): Depth level for lambda initialization.
            args (Namespace): Arguments for attention settings.
            causal (bool): Whether to use causal masking by default for self-attention.
        """
        super(DifferentialTransformerBlockrope, self).__init__()
        
        # Multihead differential attention module
        self.attn = MultiheadFlashrope(args, embed_dim, depth, num_heads)
        self.causal = causal
        self.depth = depth

        self.feed_forward = SwiGLU(embed_dim)
        self.norm1 = RMSNorm(embed_dim, eps=1e-5)
        self.norm2 = RMSNorm(embed_dim, eps=1e-5)
    
    def forward(self, x, context=None, use_cache=False,return_attn=False):
        """
        Args:
            x: Input tensor
            context: Optional context for cross-attention
            use_cache: Whether to use KV cache
        """
        # Enable/disable KV cache in attention module
        self.attn.kv_cache_enabled = use_cache

        if context is None:
            attn_out = self.attn(self.norm1(x), causal=self.causal,return_attn=return_attn)
        else:
            attn_out = self.attn(self.norm1(x), context=self.norm2(context), causal=self.causal,return_attn=return_attn)

        x = x + attn_out

        # Feed-forward network with residual connection
        ff_out = self.feed_forward(self.norm2(x))
        x = x + ff_out
        
        return x

    def init_kv_cache(self, batch_size, max_seq_len, dtype=torch.float32):
        """Initialize KV cache for this transformer block"""
        self.attn.empty_kv_cache(
            batch_size=batch_size,
            kv_cache_maxlen=max_seq_len,
            dtype=dtype
        )
        element_size = {
            torch.float32: 4,
            torch.float16: 2,
            torch.bfloat16: 2,
        }[dtype]
        
        # K cache size
        k_cache_size = (batch_size * 
                    max_seq_len * 
                    self.attn.num_heads * 
                    self.attn.head_dim * 
                    element_size)
        
        # V cache size
        v_cache_size = k_cache_size  # K和V cache大小相同
        
        # 总缓存大小
        total_cache_size = k_cache_size + v_cache_size
        cache_size_gb = total_cache_size / (1024**3)
        return cache_size_gb
    def reset_kv_cache(self):
        """Reset KV cache for this transformer block"""
        self.attn.reset_cache()

class DifferentialTransformerBlocklinearrope(nn.Module):
    def __init__(self, embed_dim, num_heads, depth, args, causal=False):
        """
        Differential Transformer Block with optional causal self-attention or cross-attention
        and automatic application of ROPE.
        
        Args:
            embed_dim (int): Embedding dimension.
            num_heads (int): Number of attention heads.
            depth (int): Depth level for lambda initialization.
            args (Namespace): Arguments for attention settings.
            causal (bool): Whether to use causal masking by default for self-attention.
        """
        super(DifferentialTransformerBlocklinearrope, self).__init__()
        
        # Multihead differential attention module
        self.attn = MultiheadFlashlinearrope(args, embed_dim, depth, num_heads)
        self.causal = causal
        self.depth = depth

        self.feed_forward = SwiGLU(embed_dim)
        self.norm1 = RMSNorm(embed_dim, eps=1e-5)
        self.norm2 = RMSNorm(embed_dim, eps=1e-5)
    
    def forward(self, x, context=None, use_cache=False):
        """
        Args:
            x: Input tensor
            context: Optional context for cross-attention
            use_cache: Whether to use KV cache
        """
        # Enable/disable KV cache in attention module
        self.attn.kv_cache_enabled = use_cache

        if context is None:
            attn_out = self.attn(self.norm1(x), causal=self.causal)
        else:
            attn_out = self.attn(self.norm1(x), context=self.norm2(context), causal=self.causal)

        x = x + attn_out

        # Feed-forward network with residual connection
        ff_out = self.feed_forward(self.norm2(x))
        x = x + ff_out
        
        return x

    def init_kv_cache(self, batch_size, max_seq_len, dtype=torch.float32):
        """Initialize KV cache for this transformer block"""
        self.attn.empty_kv_cache(
            batch_size=batch_size,
            kv_cache_maxlen=max_seq_len,
            dtype=dtype
        )
        element_size = {
            torch.float32: 4,
            torch.float16: 2,
            torch.bfloat16: 2,
        }[dtype]
        
        # K cache size
        kv_cache_size = (batch_size * 
                     self.attn.head_dim * 
                    self.attn.num_heads * 
                    self.attn.head_dim * 
                    element_size)
        

        # 总缓存大小
        total_cache_size = kv_cache_size
        cache_size_gb = total_cache_size / (1024**3)    
        return cache_size_gb
    def reset_kv_cache(self):
        """Reset KV cache for this transformer block"""
        self.attn.reset_cache()




class iFlame(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, depth=2, num_categories=0, length=10):
        super(iFlame, self).__init__()
        self.embed_dim = embed_dim
        self.depth = 2
        self.skip_weights2 = nn.Parameter(torch.ones(2))
      

        self.embedding = nn.Embedding(num_categories, embed_dim) 

        seq_len =8 * length + 16

        self.downsamplers = nn.ModuleList([
            nn.AvgPool1d(kernel_size=3, stride=3, padding=0, ceil_mode=True),
            nn.AvgPool1d(kernel_size=3, stride=3, padding=0, ceil_mode=True)
        ])
        self.upsamplers = nn.ModuleList([
            nn.Upsample(scale_factor=3, mode='nearest'),
            nn.Upsample(scale_factor=3, mode='nearest')
        ])

        self.encoder_blocks = nn.ModuleList([
            nn.ModuleList([
            DifferentialTransformerBlockrope(embed_dim, num_heads, depth=i+1, args=None, causal=True) if (i+1) % 4== 0
                else DifferentialTransformerBlocklinearrope(embed_dim, num_heads, depth=i+1, args=None, causal=True)

            for i in range(0,4)
        ]),
            nn.ModuleList([
            DifferentialTransformerBlockrope(embed_dim, num_heads, depth=i+1, args=None, causal=True) if (i+1) % 4== 0
                else DifferentialTransformerBlocklinearrope(embed_dim, num_heads, depth=i+1, args=None, causal=True)
            for i in range(4,8)
        ])
        ])


        self.bottlenecke = nn.ModuleList([
            DifferentialTransformerBlockrope(embed_dim, num_heads, depth=i+1, args=None, causal=True) if (i+1) % 4== 0
                else DifferentialTransformerBlocklinearrope(embed_dim, num_heads, depth=i+1, args=None, causal=True)
            for i in range(8,12)
        ])
        self.bottleneckd = nn.ModuleList([
            DifferentialTransformerBlockrope(embed_dim, num_heads, depth=i+1, args=None, causal=True) if (i+1) % 4== 0
                else DifferentialTransformerBlocklinearrope(embed_dim, num_heads, depth=i+1, args=None, causal=True)
            for i in range(12,16)
        ])
        self.decoder_blocks = nn.ModuleList([
            nn.ModuleList([
            DifferentialTransformerBlockrope(embed_dim, num_heads, depth=i+1, args=None, causal=True) if (i+1) % 4== 0
                else DifferentialTransformerBlocklinearrope(embed_dim, num_heads, depth=i+1, args=None, causal=True)
            for i in range(16,20)
        ]),
            nn.ModuleList([
            DifferentialTransformerBlockrope(embed_dim, num_heads, depth=i+1, args=None, causal=True) if (i+1) % 4== 0
                else DifferentialTransformerBlocklinearrope(embed_dim, num_heads, depth=i+1, args=None, causal=True)
            for i in range(20,24)
        ])
        ])

        self.output_proj = nn.Linear(embed_dim, num_categories)
        self.factor = [3, 3]
        self.norm = RMSNorm(embed_dim, eps=1e-5)
    def forward(self, x, sampled_points=None):

        if x.shape[1] % 9!= 0:
            x = pad_to_multiple(x,9, 1)
        
        x = self.embedding(x)

        x = self.norm(x) 
        encoder_outputs = []

        for scale in range(self.depth):
            for block in self.encoder_blocks[scale]:
                    x = block(x)  # Self-attention

            encoder_outputs.append(x)
            x = x.transpose(1, 2)
            x = self.downsamplers[scale](x)
            x = x.transpose(1, 2)


        for block in self.bottlenecke:

                x = block(x) 
        for i,block in enumerate(self.bottleneckd):

                x = block(x) 
                
        for scale in range(self.depth):
            x = self.upsamplers[scale](x.transpose(1, 2))
            x = x.transpose(1, 2) 
            skip = encoder_outputs[-(scale + 1)]
            


            x = shift_sequence(x, self.factor[scale] - 1)
            x =  self.skip_weights2[scale]*x + skip

            for block in self.decoder_blocks[scale]:
                    x = block(x)  # Self-attention

        x = self.output_proj(x)
        return x



    def init_kv_cache(self, batch_size, max_len=90, dtype=torch.float16):
        """初始化推理状态"""
        Gb=0
        self.use_cache = True
        self.inference_state = {
            'cache_initialized': False,
            'cur_pos': 0,
            # 'encoder_outputs': [],
            'dtype': dtype,
            'batch_size': batch_size,
            'max_len': max_len,
            # 跟踪每层中间状态
            'layer_states': {
                'encoder_0': None,  # 第一个encoder的输出
                'encoder_1': None,  # 第二个encoder的输出
                # 'bottleneck': None,  # bottleneck的输出
            },
            # 跟踪上次上采样的结果，用于跳跃连接
            'upsampled_states': {
                'decoder_0': None,
                'decoder_1': None
            }
        }
        ll=[1,3,9]
        # 初始化每个块的KV缓存
        for scale in range(self.depth):
            for i, block in enumerate(self.encoder_blocks[scale]):
                 Gb+=block.init_kv_cache(batch_size, max_len//ll[scale], dtype)
            
        for i, block in enumerate(self.bottlenecke):
            Gb+=block.init_kv_cache(batch_size, max_len//ll[2], dtype)
            
        for i, block in enumerate(self.bottleneckd):
            Gb+=block.init_kv_cache(batch_size, max_len//ll[2], dtype)
            
        for scale in range(self.depth):
            for i, block in enumerate(self.decoder_blocks[scale]):
                Gb+=block.init_kv_cache(batch_size, max_len//ll[1-scale], dtype)
        return Gb
    
    def reset_kv_cache(self):
        """重置推理状态"""
        if hasattr(self, 'inference_state'):
            # 重置状态
            self.inference_state['cur_pos'] = 0
            # self.inference_state['encoder_outputs'] = []
            self.inference_state['cache_initialized'] = False
            self.inference_state['layer_states'] = {
                'encoder_0': None,
                'encoder_1': None,
                'bottleneck': None,
            }
            self.inference_state['upsampled_states'] = {
                'decoder_0': None,
                'decoder_1': None
            }
            
            # 重置每个块的KV缓存
            for scale in range(self.depth):
                for block in self.encoder_blocks[scale]:
                    block.reset_kv_cache()
                
            for block in self.bottlenecke:
                block.reset_kv_cache()
                
            for block in self.bottleneckd:
                block.reset_kv_cache()
                
            for scale in range(self.depth):
                for block in self.decoder_blocks[scale]:
                    block.reset_kv_cache()
    
    def _process_first_tokens(self, x):
        """处理初始token序列，构建初始缓存状态"""

        x = self.embedding(x)
        x = self.norm(x)
        
        # === Encoder阶段 ===
        encoder_outputs = []
        
        # Encoder层0 (A)
        for block in self.encoder_blocks[0]:
            x = block(x, use_cache=True)
        encoder_outputs.append(x)
        self.inference_state['layer_states']['encoder_0'] = x[:, -3:]
        
        # 下采样
        x_downsampled = x.transpose(1, 2)
        x_downsampled = self.downsamplers[0](x_downsampled)
        x_downsampled = x_downsampled.transpose(1, 2)
        
        # Encoder层1 (B)
        for block in self.encoder_blocks[1]:
            x_downsampled = block(x_downsampled, use_cache=True)
        encoder_outputs.append(x_downsampled)
        self.inference_state['layer_states']['encoder_1'] = x_downsampled[:, -3:]
        
        # 再次下采样
        x_bottleneck = x_downsampled.transpose(1, 2)
        x_bottleneck = self.downsamplers[1](x_bottleneck)
        x_bottleneck = x_bottleneck.transpose(1, 2)
        
        # Bottleneck层 (C)
        for block in self.bottlenecke:
            x_bottleneck = block(x_bottleneck, use_cache=True)
            
        for block in self.bottleneckd:
            x_bottleneck = block(x_bottleneck, use_cache=True)
        
        # self.inference_state['layer_states']['bottleneck'] = x_bottleneck
        
        # === Decoder阶段 ===
        # Decoder层0 (D)
        x_upsampled = self.upsamplers[0](x_bottleneck.transpose(1, 2)).transpose(1, 2)
        self.inference_state['upsampled_states']['decoder_0'] = x_upsampled[:, -3:]
        
        skip = encoder_outputs[1]  # 第二个encoder输出
        
        x_upsampled = shift_sequence(x_upsampled, self.factor[0] - 1)
        x_upsampled = self.skip_weights2[0] * x_upsampled + skip
        
        for block in self.decoder_blocks[0]:
            x_upsampled = block(x_upsampled, use_cache=True)
        
        # Decoder层1 (E)
        x_final = self.upsamplers[1](x_upsampled.transpose(1, 2)).transpose(1, 2)
        self.inference_state['upsampled_states']['decoder_1'] = x_final[:, -3:]
        
        skip = encoder_outputs[0]  # 第一个encoder输出
        
        x_final = shift_sequence(x_final, self.factor[1] - 1)
        x_final = self.skip_weights2[1] * x_final + skip
        
        for block in self.decoder_blocks[1]:
            x_final = block(x_final, use_cache=True)
        
        # 最终输出投影
        logits = self.output_proj(x_final)
        
        # 更新状态
        # self.inference_state['encoder_outputs'] = encoder_outputs
        self.inference_state['cur_pos'] = x.shape[1]
        self.inference_state['cache_initialized'] = True
        
        return logits
    def _process_single_token(self, x):
        """处理单个token的推理"""
        batch_size = x.shape[0]
        cur_pos = self.inference_state['cur_pos']
        
        # 词嵌入和规范化
        x = self.embedding(x)
        x = self.norm(x)
        
        # === 选择性更新策略 ===
        update_encoder_0 = True  # 始终更新第一个encoder (A)
        update_encoder_1 = (cur_pos + 1) % 3 == 0  # 当(n+1)是3的倍数时更新
        update_bottleneck = (cur_pos + 1) % 9 == 0  # 当(n+1)是9的倍数时更新
        update_decoder_0 = (cur_pos + 1) % 3 == 0  # 当(n+1)是3的倍数时更新
        update_decoder_1 = True  # 始终更新最后一个decoder (E)
        
        # === Encoder阶段 (只处理单个token) ===
        # Encoder层0 (A)
        if update_encoder_0:
            encoder_0_output = x
            for block in self.encoder_blocks[0]:
                encoder_0_output = block(encoder_0_output, use_cache=True)
            self.inference_state['layer_states']['encoder_0'][:,cur_pos% 3:cur_pos% 3+1] = encoder_0_output
            # 追加到当前序列，而不是替换
            # if self.inference_state['layer_states']['encoder_0'] is not None:
            #     self.inference_state['layer_states']['encoder_0'] = torch.cat(
            #         [self.inference_state['layer_states']['encoder_0'], encoder_0_output], dim=1
            #     )
            # else:
            #     self.inference_state['layer_states']['encoder_0'] = encoder_0_output
            
        # 如果需要更新第二层encoder
        if update_encoder_1:
            # 下采样时只考虑最近3个token（包括当前token）
            recent_tokens = 3
            # 获取最近的token的encoder_0输出
            recent_encoder_outputs = self.inference_state['layer_states']['encoder_0']#[:, (cur_pos-2)% 3:(cur_pos)%3+1 ]

            
            # 下采样
            x_downsampled = recent_encoder_outputs.transpose(1, 2)
            x_downsampled = self.downsamplers[0](x_downsampled)
            x_downsampled = x_downsampled.transpose(1, 2)
            
            # 第二层encoder处理
            for block in self.encoder_blocks[1]:
                x_downsampled = block(x_downsampled, use_cache=True)
            self.inference_state['layer_states']['encoder_1'][:,(cur_pos-2)%9//3:(cur_pos-2)%9//3+1] = x_downsampled
            # 追加到当前序列，而不是替换
            # if self.inference_state['layer_states']['encoder_1'] is not None:
            #     self.inference_state['layer_states']['encoder_1'] = torch.cat(
            #         [self.inference_state['layer_states']['encoder_1'], x_downsampled], dim=1
            #     )
            # else:
            #     self.inference_state['layer_states']['encoder_1'] = x_downsampled
        
        # 如果需要更新bottleneck
        if update_bottleneck:
            # 再次下采样
            recent_tokens = 3
            # 获取最近的token的encoder_1输出
            # if cur_pos > 2:  # 确保有足够的token
            #     recent_encoder_1_outputs = self.inference_state['layer_states']['encoder_1'][:, -recent_tokens:]
            # else:
            recent_encoder_1_outputs = self.inference_state['layer_states']['encoder_1']#[:, -recent_tokens:]
            
            x_bottleneck = recent_encoder_1_outputs.transpose(1, 2)
            x_bottleneck = self.downsamplers[1](x_bottleneck)
            x_bottleneck = x_bottleneck.transpose(1, 2)
            
            # Bottleneck层处理
            for block in self.bottlenecke:
                x_bottleneck = block(x_bottleneck, use_cache=True)
            
            for block in self.bottleneckd:
                x_bottleneck = block(x_bottleneck, use_cache=True)
            
            # # 追加到当前序列，而不是替换
            # if self.inference_state['layer_states']['bottleneck'] is not None:
            #     self.inference_state['layer_states']['bottleneck'] = torch.cat(
            #         [self.inference_state['layer_states']['bottleneck'], x_bottleneck], dim=1
            #     )
            # else:
            #     self.inference_state['layer_states']['bottleneck'] = x_bottleneck
                
            # bottleneck更新后立即执行第一层上采样
            bottleneck_output =x_bottleneck# self.inference_state['layer_states']['bottleneck']
            x_upsampled = self.upsamplers[0](bottleneck_output.transpose(1, 2)).transpose(1, 2)
            self.inference_state['upsampled_states']['decoder_0'] = x_upsampled
            # 保存上采样结果
            # if self.inference_state['upsampled_states']['decoder_0'] is not None:
            #     self.inference_state['upsampled_states']['decoder_0'] = torch.cat(
            #         [self.inference_state['upsampled_states']['decoder_0'], x_upsampled], dim=1
            #     )
            # else:
            #     self.inference_state['upsampled_states']['decoder_0'] = x_upsampled
        
        # === Decoder阶段 ===
        # 如果需要更新第一个decoder
        if update_decoder_0:
            # 计算正确的上采样状态索引
            # bottleneck每9个token更新一次，所以decoder_0的upsampled结果也是
            upsampled_decoder0_idx = ((cur_pos-2)%9//3-2)%3  # 第一个上采样结果的索引
            
            # 使用正确索引获取上采样结果
            x_upsampled = self.inference_state['upsampled_states']['decoder_0'][:, upsampled_decoder0_idx:upsampled_decoder0_idx+1]
            
 
            # 获取对应位置的encoder_1输出作为跳跃连接
            # 使用基于cur_pos的索引来选择正确的token
            encoder_1_skip_idx =  (cur_pos-2)%9//3   # 因为encoder_1是每3个token更新一次
            encoder_1_output = self.inference_state['layer_states']['encoder_1'][:, encoder_1_skip_idx:encoder_1_skip_idx+1]
            
            # 应用跳跃连接
            x_upsampled = self.skip_weights2[0] * x_upsampled + encoder_1_output
            
            # 第一个decoder处理
            for block in self.decoder_blocks[0]:
                x_upsampled = block(x_upsampled, use_cache=True)
            
            # decoder_0更新后立即执行第二层上采样
            x_final = self.upsamplers[1](x_upsampled.transpose(1, 2)).transpose(1, 2)
            
            # 保存第二层上采样结果
            # if self.inference_state['upsampled_states']['decoder_1'] is not None:
            #     self.inference_state['upsampled_states']['decoder_1'] = torch.cat(
            #         [self.inference_state['upsampled_states']['decoder_1'], x_final], dim=1
            #     )
            # else:
            #     self.inference_state['upsampled_states']['decoder_1'] = x_final
            self.inference_state['upsampled_states']['decoder_1'] = x_final
        # else:
        #     # 使用之前处理好的结果，避免重复计算
        #     # 计算正确的decoder_0位置索引 (每3个token更新一次)
        #     decoder0_idx = (cur_pos // 3) - 1 if cur_pos >= 3 else 0
        #     # 确保索引在有效范围内
        #     decoder0_idx = max(0, min(decoder0_idx, self.inference_state['upsampled_states']['decoder_0'].shape[1] - 1))
            
        #     x_upsampled = self.inference_state['upsampled_states']['decoder_0'][:, decoder0_idx:decoder0_idx+1]
            
        #     # 当不更新decoder_0时，直接通过block保持KV缓存更新
        #     for block in self.decoder_blocks[0]:
        #         x_upsampled = block(x_upsampled, use_cache=True)
        
        # 计算正确的第二层上采样结果索引 (每3个token更新一次，与decoder_0同步)
        if update_decoder_1:
            upsampled_decoder1_idx =  (cur_pos-2)%3

            x_final = self.inference_state['upsampled_states']['decoder_1'][:, upsampled_decoder1_idx:upsampled_decoder1_idx+1]
  
            encoder_0_output = self.inference_state['layer_states']['encoder_0'][:, cur_pos % 3:cur_pos % 3+1]

            x_final = self.skip_weights2[1] * x_final + encoder_0_output
            
            # 最后一个decoder处理
            for block in self.decoder_blocks[1]:
                x_final = block(x_final, use_cache=True)
            
        # 最终输出投影
        logits = self.output_proj(x_final)
        
        # 更新位置
        self.inference_state['cur_pos'] += 1
        
        return logits
    def inference_step(self, x,pc=None, use_cache=True):
        """
        执行推理步骤，根据输入是序列还是单个token调用不同处理函数
        """
        # 确保已初始化
        # self.cache_size=self.init_kv_cache(batch_size, max_seq_len, dtype=torch.float16)
        
        if not hasattr(self, 'inference_state'):
            self.cache_size=self.init_inference(x.shape[0])
        
        # 根据是否已初始化缓存选择处理函数
        if not self.inference_state['cache_initialized']:
            return self._process_first_tokens(x)
        else:
            # 确保输入是单个token
            if x.shape[1] > 1:
                # 如果输入多个token，逐个处理
                all_logits = []
                for i in range(x.shape[1]):
                    token = x[:, i:i+1]
                    logits = self._process_single_token(token)
                    all_logits.append(logits)
                # 连接所有结果
                return torch.cat(all_logits, dim=1)
            else:
                return self._process_single_token(x)
    @torch.no_grad()
    def generate_sequence(
        self,
        initial_input: torch.Tensor,
        pc,
        max_seq_len: int,
        device: str,
        shorten_factor: int = 3,
        end_symbol: int = 129,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.95,
        temperature: float = 1.0
    ) -> np.ndarray:
        """
        Generate sequence using the same interface as the original generate_sequence function
        """
        self.eval()
        generated = initial_input.to(device)
        
        # Initialize KV cache
        batch_size = generated.size(0)
        self.cache_size=self.init_kv_cache(batch_size, max_seq_len, dtype=torch.float16)
        
        with torch.no_grad():
            # First forward pass with the entire initial sequence
            with torch.cuda.amp.autocast():
                output = self.inference_step(generated, pc, use_cache=True)
            
            # Then generate one token at a time
            for _ in range(max_seq_len - generated.size(1)):
                # Get logits for the last position
                last_logits = output[:, -1, :]
                
                # Apply temperature
                logits = last_logits / temperature
                
                # Calculate probability distribution
                probs = F.softmax(logits, dim=-1)
                
                # Apply Top-K filtering
                if top_k is not None and top_k > 0:
                    top_k = min(top_k, probs.size(-1))
                    topk_probs, topk_indices = torch.topk(probs, top_k, dim=-1)
                    mask = torch.zeros_like(probs, dtype=torch.bool)
                    mask.scatter_(1, topk_indices, 1)
                    probs = probs.masked_fill(~mask, 0.0)
                
                # Apply Top-P filtering
                if top_p is not None and top_p > 0.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = 0
                    
                    sorted_probs = sorted_probs.masked_fill(sorted_indices_to_remove, 0.0)
                    probs = torch.zeros_like(probs).scatter_(1, sorted_indices, sorted_probs)
                
                # Sample next token
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=1)

                # Forward pass with only the new token
                with torch.cuda.amp.autocast():
                    output = self.inference_step(next_token, pc, use_cache=True)

        # Reset cache after generation
        self.reset_kv_cache()
        return generated.cpu().numpy()
    
