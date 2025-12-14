import math

import torch
from torch import nn
import einx
from jaxtyping import Num, Integer, Float, Bool, Int

class Linear(nn.Module):

    def __init__(
        self, 
        in_features:int, 
        out_features:int, 
        device:torch.device | None=None, 
        dtype:torch.dtype | None=None
    ):
        super().__init__()
        weights = torch.empty((out_features, in_features), device=device, dtype=dtype)
        std_dev = math.sqrt(2/(in_features+out_features))
        mean = 0
        weights = nn.init.trunc_normal_(weights, mean, std_dev, a=-3*std_dev, b=3*std_dev)
        self.w = nn.Parameter(weights, requires_grad=True)

    def forward(self, x:Num[torch.Tensor, "... d_in"]) -> Num[torch.Tensor, "... d_out"]:
        return einx.dot("d_out [d_in], b... [d_in] -> b... d_out", self.w, x)

class Embedding(nn.Module):
    def __init__(
        self, 
        num_embeddings:int, 
        embedding_dim: int, 
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        weights = torch.empty((num_embeddings, embedding_dim), dtype=dtype, device=device)
        weights = nn.init.trunc_normal_(weights, mean=0, std=1, a=-3, b=3)
        self.embed_mat = nn.Parameter(weights, requires_grad=True)

    def forward(self, token_ids: Integer[torch.Tensor, "batch_size seq_len"]) -> Integer[torch.Tensor, "batch_size seq_len d_model"]:
        # unsqueeze, then perform lookup on embedding matrix
        return einx.get_at("[vocab] d_model, batch_size (seq_len [1]) -> batch_size seq_len d_model", self.embed_mat, token_ids)


class RMSNorm(nn.Module):
    def __init__(self, d_model:int, eps:float=1e-5, device:torch.device|None=None, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        gain = torch.ones((d_model,), dtype=dtype, device=device)
        self.g = nn.Parameter(gain, requires_grad=True)
        self.eps:float = eps
        self.d_model:int = d_model


    def forward(self, x: Num[torch.Tensor, "batch_size seq_len d_model"]) ->  Num[torch.Tensor, "batch_size seq_len d_model"]:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt((torch.pow(x, 2) + self.eps).sum(-1)/self.d_model)
        
        # x.shape = (batch, seq_len, d_model)
        # rms.shape = (batch, seq_len)
        # g.shape = (d_model, )
        result = (x / rms.unsqueeze(-1)) * self.g

        return result.to(in_dtype)

# SiLU/Swish activation function
def silu(x:torch.Tensor):
    return x * torch.sigmoid(x)

# Combination between SiLU and Gated Linear Units
class SwiGLU(nn.Module):
    def __init__(self, d_model:int, d_ff:int | None = None, device:torch.device | None = None, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        if not d_ff:
            d_ff:int = round(((8/3)*d_model)/64)
            d_ff *= 64 # ensure that d_ff is multiple of 64
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        assert d_k%2==0, "d_k must be divisible by 2 to use RoPE"

        multiplier = torch.arange(max_seq_len).float()
        angle = theta ** (-2*torch.arange(d_k//2).float()/d_k)
        
        # broadast: (max_seq_len x 1) x (1 x d_k_half) = (max_seq_len x d_k_half)
        final_angle = einx.multiply('max_seq_len, d_k_half -> max_seq_len d_k_half', multiplier, angle).to(device=device)
        
        # persistent=False ensures that this tensor is not part of state_dict, which is okay since we can always recompute it.
        self.register_buffer('cos_buf', torch.cos(final_angle), persistent=False)
        self.register_buffer('sin_buf', torch.sin(final_angle), persistent=False)
        

    def forward(
        self,
        x: Num[torch.Tensor, "... seq_len d_k"],
        token_positions: Num[torch.Tensor, "... seq_len"]
    ) -> Num[torch.Tensor, "... seq_len d_k"]:  
        
        # index the cos and sin that we are interested in
        cos_buf = einx.get_at("[max_seq_len] d_k_half, b... (seq_len [1]) -> b... seq_len d_k_half", self.cos_buf, token_positions)
        sin_buf = einx.get_at("[max_seq_len] d_k_half, b... (seq_len [1]) -> b... seq_len d_k_half", self.sin_buf, token_positions)
        
        x = einx.rearrange("b... seq_len (d_k_half a) -> b... seq_len d_k_half a", x, a=2)
        
        x_even = x[..., 0]
        x_odd = x[..., 1]

        x_even_new = (cos_buf * x_even) - (sin_buf * x_odd)
        x_odd_new = (cos_buf * x_odd) + (sin_buf * x_even)
    
        out = torch.zeros_like(x)
        out[..., 0] = x_even_new
        out[..., 1] = x_odd_new
        
        x = einx.rearrange("b... seq_len d_k_half a -> b... seq_len (d_k_half a)", out, a=2)
        return x

def softmax(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    x = tensor
    safe_x = x - torch.max(x, dim=dim, keepdim=True).values
    exp_x = torch.exp(safe_x)
    return exp_x/torch.sum(exp_x, dim=dim, keepdim=True)

def scaled_dot_product_attention(
    Q: Float[torch.Tensor, "batch_size ... seq_len_q d_k"],
    K: Float[torch.Tensor, "batch_size ... seq_len_k d_k"],
    V: Float[torch.Tensor, "batch_size ... seq_len_k d_v"],
    mask: Bool[torch.Tensor, "batch_size ... seq_len_q seq_len_k"] | None = None,
) -> Float[torch.Tensor, "batch_size ... d_v"]:
    assert Q.dtype==K.dtype==V.dtype==torch.float
    # Computes QK^T / d_k^0.5
    scaled_qk_t = einx.dot("b... seq_len_q d_k, b... seq_len_k d_k -> b... seq_len_q seq_len_k", Q, K) / math.sqrt(Q.shape[-1])

    to_add = torch.zeros_like(mask, dtype=torch.float)
    to_add[~mask] = float('-inf')

    scaled_qk_t_masked = einx.add("b... seq_len_q seq_len_k, b... seq_len_q seq_len_k -> b... seq_len_q seq_len_k", scaled_qk_t, to_add)

    softmaxed_weights = softmax(scaled_qk_t_masked, dim=-1)
    
    return einx.dot("b... seq_len_q seq_len_k, b... seq_len_k d_v -> b... seq_len_q d_v", softmaxed_weights, V)


class MultiheadSelfAttention(nn.Module):
    def __init__(
        self, 
        d_model:int, 
        num_heads:int, 
        dtype: torch.dtype | None=None,
        device:torch.device | None=None
    ) -> None:
        super().__init__()
        assert d_model%num_heads==0, "d_model must be a multiplier of num_heads!"
        self.o_weight = Linear(d_model, d_model, device=device, dtype=dtype)
        self.d_model = d_model
        self.num_heads = num_heads
        self.qkv_weights = Linear(d_model, 3*d_model, device=device, dtype=dtype)
    
    def compute_mhsa(
        self,
        Q: Float[torch.Tensor, "batch_size head seq_len_q d_k"],
        K: Float[torch.Tensor, "batch_size head seq_len_k d_k"],
        V: Float[torch.Tensor, "batch_size head seq_len_k d_v"],
    )-> Float[torch.Tensor, "batch_size seq_len d_model"]:
         # batch_size x head x seq_len_q x seq_len_k
        mask = torch.ones(Q.shape[:-1]+(K.shape[-2],))
        mask = torch.tril(mask).bool() # set upper triangular part to False

        attn = scaled_dot_product_attention(Q, K, V, mask=mask)
        attn = einx.rearrange("batch head seq_len_q dv -> batch seq_len_q (head dv)", attn, head=self.num_heads)
        return self.o_weight(attn)

    def forward(
        self,
        x: Float[torch.Tensor, " ... sequence_length d_model"]
    ) -> Float[torch.Tensor, " ... sequence_length d_model"]:
        Q,K,V = einx.rearrange("b... seq_len (o head d_k) -> o b... head seq_len d_k", self.qkv_weights(x), head=self.num_heads, o=3)
        return self.compute_mhsa(Q,K,V)

       
class MultiheadSelfAttentionWithRoPE(MultiheadSelfAttention):
    def __init__(
        self, 
        d_model:int, 
        num_heads:int, 
        theta: float,
        max_seq_len: int,
        device:torch.device | None=None,
        dtype: torch.dtype | None = None
    ) -> None:
        super().__init__(d_model=d_model, num_heads=num_heads, device=device, dtype=dtype)
        d_k = d_model//num_heads
        self.rope = RotaryPositionalEmbedding(theta, d_k, max_seq_len)

    def forward(
        self,
        x: Float[torch.Tensor, " ... sequence_length d_model"],
        token_positions: Int[torch.Tensor, " ... sequence_length"] | None = None,
    ) -> Float[torch.Tensor, " ... sequence_length d_model"]:
        Q,K,V = einx.rearrange("b... seq_len (o head d_k) -> o b... head seq_len d_k", self.qkv_weights(x), head=self.num_heads, o=3)

        if token_positions is None:
            # use broadcasting: (batch head 1) x (1 seq_len) to get (batch head seq_len)
            token_positions = torch.arange(x.shape[-2])
            broadcast_tensor = torch.ones(Q.shape[:-2]).long()
            token_positions = einx.multiply("b... head, seq_len -> b... head seq_len", broadcast_tensor, token_positions)
        else:
            # use repeat: (batch seq_len) -> (head batch seq_len), then rearrange to get (batch head seq_len) 
            token_positions = token_positions.repeat(self.num_heads, 1, 1)
            token_positions = einx.rearrange("head batch seq_len -> batch head seq_len", token_positions)
        
        rotated_q = self.rope(Q, token_positions)
        rotated_k = self.rope(K, token_positions)

        return self.compute_mhsa(rotated_q, rotated_k, V)


class PreNormTransformerBlock(nn.Module):
    def __init__(
        self, 
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.attn = MultiheadSelfAttentionWithRoPE(d_model, num_heads, theta, max_seq_len, device=device, dtype=dtype)
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

        
    def forward(
        self,
        x: Float[torch.Tensor, "batch_size seq_len d_model"],
        token_positions: Int[torch.Tensor, " ... sequence_length"] | None = None,
    )->Float[torch.Tensor, "batch_size seq_len d_model"]:
        ln1_out = self.ln1(x)
        attn_out = self.attn(ln1_out, token_positions) 
        sublayer_1_out = attn_out + x

        ln2_out = self.ln2(sublayer_1_out)
        ffn_out = self.ffn(ln2_out)

        return ffn_out + sublayer_1_out