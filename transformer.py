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
        v_cache_size = k_cache_size  
        
       
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
     
        Gb=0
        self.use_cache = True
        self.inference_state = {
            'cache_initialized': False,
            'cur_pos': 0,
            # 'encoder_outputs': [],
            'dtype': dtype,
            'batch_size': batch_size,
            'max_len': max_len,
          
            'layer_states': {
                'encoder_0': None,  
                'encoder_1': None,  
                # 'bottleneck': None,  
            },
    
            'upsampled_states': {
                'decoder_0': None,
                'decoder_1': None
            }
        }
        ll=[1,3,9]
    
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
   
        if hasattr(self, 'inference_state'):
          
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

        x = self.embedding(x)
        x = self.norm(x)
        

        encoder_outputs = []
        

        for block in self.encoder_blocks[0]:
            x = block(x, use_cache=True)
        encoder_outputs.append(x)
        self.inference_state['layer_states']['encoder_0'] = x[:, -3:]
        

        x_downsampled = x.transpose(1, 2)
        x_downsampled = self.downsamplers[0](x_downsampled)
        x_downsampled = x_downsampled.transpose(1, 2)
        

        for block in self.encoder_blocks[1]:
            x_downsampled = block(x_downsampled, use_cache=True)
        encoder_outputs.append(x_downsampled)
        self.inference_state['layer_states']['encoder_1'] = x_downsampled[:, -3:]
        

        x_bottleneck = x_downsampled.transpose(1, 2)
        x_bottleneck = self.downsamplers[1](x_bottleneck)
        x_bottleneck = x_bottleneck.transpose(1, 2)
        

        for block in self.bottlenecke:
            x_bottleneck = block(x_bottleneck, use_cache=True)
            
        for block in self.bottleneckd:
            x_bottleneck = block(x_bottleneck, use_cache=True)
        
  
        x_upsampled = self.upsamplers[0](x_bottleneck.transpose(1, 2)).transpose(1, 2)
        self.inference_state['upsampled_states']['decoder_0'] = x_upsampled[:, -3:]
        
        skip = encoder_outputs[1]  
        
        x_upsampled = shift_sequence(x_upsampled, self.factor[0] - 1)
        x_upsampled = self.skip_weights2[0] * x_upsampled + skip
        
        for block in self.decoder_blocks[0]:
            x_upsampled = block(x_upsampled, use_cache=True)
        

        x_final = self.upsamplers[1](x_upsampled.transpose(1, 2)).transpose(1, 2)
        self.inference_state['upsampled_states']['decoder_1'] = x_final[:, -3:]
        
        skip = encoder_outputs[0]  
        
        x_final = shift_sequence(x_final, self.factor[1] - 1)
        x_final = self.skip_weights2[1] * x_final + skip
        
        for block in self.decoder_blocks[1]:
            x_final = block(x_final, use_cache=True)
        

        logits = self.output_proj(x_final)
        

        self.inference_state['cur_pos'] = x.shape[1]
        self.inference_state['cache_initialized'] = True
        
        return logits
    def _process_single_token(self, x):

        batch_size = x.shape[0]
        cur_pos = self.inference_state['cur_pos']
        

        x = self.embedding(x)
        x = self.norm(x)
        
 
        update_encoder_0 = True  
        update_encoder_1 = (cur_pos + 1) % 3 == 0 
        update_bottleneck = (cur_pos + 1) % 9 == 0 
        update_decoder_0 = (cur_pos + 1) % 3 == 0  
        update_decoder_1 = True  
   
        if update_encoder_0:
            encoder_0_output = x
            for block in self.encoder_blocks[0]:
                encoder_0_output = block(encoder_0_output, use_cache=True)
            self.inference_state['layer_states']['encoder_0'][:,cur_pos% 3:cur_pos% 3+1] = encoder_0_output
          
        if update_encoder_1:
            
            recent_tokens = 3
      
            recent_encoder_outputs = self.inference_state['layer_states']['encoder_0']#[:, (cur_pos-2)% 3:(cur_pos)%3+1 ]

            x_downsampled = recent_encoder_outputs.transpose(1, 2)
            x_downsampled = self.downsamplers[0](x_downsampled)
            x_downsampled = x_downsampled.transpose(1, 2)
            
          
            for block in self.encoder_blocks[1]:
                x_downsampled = block(x_downsampled, use_cache=True)
            self.inference_state['layer_states']['encoder_1'][:,(cur_pos-2)%9//3:(cur_pos-2)%9//3+1] = x_downsampled
     
        if update_bottleneck:
            
            recent_tokens = 3
      
            recent_encoder_1_outputs = self.inference_state['layer_states']['encoder_1']#[:, -recent_tokens:]
            
            x_bottleneck = recent_encoder_1_outputs.transpose(1, 2)
            x_bottleneck = self.downsamplers[1](x_bottleneck)
            x_bottleneck = x_bottleneck.transpose(1, 2)
            

            for block in self.bottlenecke:
                x_bottleneck = block(x_bottleneck, use_cache=True)
            
            for block in self.bottleneckd:
                x_bottleneck = block(x_bottleneck, use_cache=True)
            
   
            bottleneck_output =x_bottleneck# self.inference_state['layer_states']['bottleneck']
            x_upsampled = self.upsamplers[0](bottleneck_output.transpose(1, 2)).transpose(1, 2)
            self.inference_state['upsampled_states']['decoder_0'] = x_upsampled

        if update_decoder_0:
           
     
            upsampled_decoder0_idx = ((cur_pos-2)%9//3-2)%3 
            
     
            x_upsampled = self.inference_state['upsampled_states']['decoder_0'][:, upsampled_decoder0_idx:upsampled_decoder0_idx+1]
            
 
         
            encoder_1_skip_idx =  (cur_pos-2)%9//3  
            encoder_1_output = self.inference_state['layer_states']['encoder_1'][:, encoder_1_skip_idx:encoder_1_skip_idx+1]
            
  
            x_upsampled = self.skip_weights2[0] * x_upsampled + encoder_1_output
            

            for block in self.decoder_blocks[0]:
                x_upsampled = block(x_upsampled, use_cache=True)
            

            x_final = self.upsamplers[1](x_upsampled.transpose(1, 2)).transpose(1, 2)
            

            self.inference_state['upsampled_states']['decoder_1'] = x_final
       
        if update_decoder_1:
            upsampled_decoder1_idx =  (cur_pos-2)%3

            x_final = self.inference_state['upsampled_states']['decoder_1'][:, upsampled_decoder1_idx:upsampled_decoder1_idx+1]
  
            encoder_0_output = self.inference_state['layer_states']['encoder_0'][:, cur_pos % 3:cur_pos % 3+1]

            x_final = self.skip_weights2[1] * x_final + encoder_0_output
            
            for block in self.decoder_blocks[1]:
                x_final = block(x_final, use_cache=True)
            
       
        logits = self.output_proj(x_final)
        
      
        self.inference_state['cur_pos'] += 1
        
        return logits
    def inference_step(self, x,pc=None, use_cache=True):
      
        # self.cache_size=self.init_kv_cache(batch_size, max_seq_len, dtype=torch.float16)
        
        if not hasattr(self, 'inference_state'):
            self.cache_size=self.init_inference(x.shape[0])
        
      
        if not self.inference_state['cache_initialized']:
            return self._process_first_tokens(x)
        else:
           
            if x.shape[1] > 1:
                #
                all_logits = []
                for i in range(x.shape[1]):
                    token = x[:, i:i+1]
                    logits = self._process_single_token(token)
                    all_logits.append(logits)
               
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
    
