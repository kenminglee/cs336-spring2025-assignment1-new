import math
from collections.abc import Callable, Iterable
import typing
import os

import torch
from torch import nn
import einx
from jaxtyping import Num, Integer, Float, Bool, Int
from torch.optim.optimizer import ParamsT
from tqdm import tqdm

from cs336_basics.bpe.tokenization import Tokenizer

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

        out = torch.stack(
            [
                (cos_buf * x_even) - (sin_buf * x_odd), 
                (cos_buf * x_odd) + (sin_buf * x_even)
            ], 
            dim=-1
        )
        
        x = einx.rearrange("b... seq_len d_k_half a -> b... seq_len (d_k_half a)", out, a=2)
        return x

def softmax(tensor: torch.Tensor, dim: int, temperature:float=1.0) -> torch.Tensor:
    x = tensor
    safe_x = x - torch.max(x, dim=dim, keepdim=True).values
    exp_x = torch.exp(safe_x/temperature)
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
        mask = torch.ones(Q.shape[:-1]+(K.shape[-2],), device=Q.device)
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
        self.rope = RotaryPositionalEmbedding(theta, d_k, max_seq_len, device=device)

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
        max_seq_len: int,
        theta: float,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.attn = MultiheadSelfAttentionWithRoPE(d_model, num_heads, theta, max_seq_len, device=device, dtype=dtype)
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff=d_ff, device=device, dtype=dtype)

        
    def forward(
        self,
        x: Float[torch.Tensor, "batch_size seq_len d_model"],
        token_positions: Int[torch.Tensor, " ... seq_len"] | None = None,
    )->Float[torch.Tensor, "batch_size seq_len d_model"]:
        ln1_out = self.ln1(x)
        attn_out = self.attn(ln1_out, token_positions) 
        sublayer_1_out = attn_out + x

        ln2_out = self.ln2(sublayer_1_out)
        ffn_out = self.ffn(ln2_out)
        sublayer_2_out = ffn_out + sublayer_1_out

        return sublayer_2_out

class PostNormTransformerBlock(nn.Module):
    def __init__(
        self, 
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        theta: float,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.attn = MultiheadSelfAttentionWithRoPE(d_model, num_heads, theta, max_seq_len, device=device, dtype=dtype)
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff=d_ff, device=device, dtype=dtype)

        
    def forward(
        self,
        x: Float[torch.Tensor, "batch_size seq_len d_model"],
        token_positions: Int[torch.Tensor, " ... seq_len"] | None = None,
    )->Float[torch.Tensor, "batch_size seq_len d_model"]:
        attn_out = self.attn(x, token_positions) 
        sublayer_1_out = self.ln1(attn_out + x)

        ffn_out = self.ffn(sublayer_1_out)
        sublayer_2_out = self.ln2(ffn_out + sublayer_1_out)

        return sublayer_2_out

class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int, # determines max-seq-len
        num_layers: int, # number of transformer blocks
        d_model: int,
        num_heads: int,
        rope_theta: float, # constant for RoPE
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        self.context_length = context_length
        self.token_embeddings = Embedding(num_embeddings=vocab_size, embedding_dim=d_model, device=device, dtype=dtype)
        layers = []
        for _ in range(num_layers):
            layers.append(PreNormTransformerBlock(
                d_model = d_model,
                num_heads = num_heads,
                max_seq_len = context_length,
                theta = rope_theta,
                d_ff = d_ff,
                device = device,
                dtype = dtype
            ))
        self.layers = nn.ModuleList(layers) # ensures that all transformer blocks are properly registered
        self.ln_final = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(
        self,
        token_ids: Int[torch.Tensor, "batch_size seq_len"],
        token_positions: Int[torch.Tensor, "batch_size seq_len"] | None = None,
    )->Float[torch.Tensor, "batch_size seq_len vocab_size"]:
        assert token_ids.shape[-1]<=self.context_length, "Input is too long!"
        # x.shape = (batch_size seq_len d_model)
        x = self.token_embeddings(token_ids)
        for transformer_block in self.layers:
            x = transformer_block(x, token_positions)
        x = self.ln_final(x)
        # x.shape after lm_head: (batch_size seq_len vocab)
        x = self.lm_head(x)
        return x
        
def cross_entropy_loss(
    logits: Float[torch.Tensor, "batch_size ... vocab_size"],
    target_indices: Int[torch.Tensor, "batch_size ..."]
)-> Float[torch.Tensor, ""]:
    
    # flatten if needed
    logits = einx.rearrange("batch_size ... vocab -> (batch_size ...) vocab", logits)
    target_indices = einx.rearrange("batch_size ... -> (batch_size ...)", target_indices)

    logits = logits - torch.max(logits, dim=-1, keepdim=True).values
    logits_logsumexp = torch.log(logits.exp().sum(dim=-1))

    # log cancels out exp term, hence a simple lookup
    logits_at_target = einx.get_at("batch_size [vocab], (batch_size [1]) -> batch_size", logits, target_indices)
    
    return (logits_logsumexp - logits_at_target).mean()


class AdamW(torch.optim.Optimizer):
    def __init__(
        self, 
        params: ParamsT, 
        lr: float,
        weight_decay: float,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
    ) -> None:
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {
            "lr":lr,
            "lambda":weight_decay,
            "beta_1":betas[0],
            "beta_2":betas[1],
            "epsilon":eps
        }
        super().__init__(params, defaults)
    
    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            alpha, lamb, beta_1, beta_2, eps = group["lr"], group["lambda"], group["beta_1"], group["beta_2"], group["epsilon"] # fetch hyperparameters
            for p in group["params"]:
                if p.grad is None: 
                    continue
                # Get state for this parameter
                state = self.state[p]
                # Get the gradient of loss with respect to p.
                grad = p.grad.data 
                # Update timestep
                t = state["t"] = state.get("t", 0) + 1
                # Update first moment estimate
                m = state["m"] =  beta_1 * state.get("m", torch.zeros_like(p)) + (1-beta_1)*grad
                # Update second moment estimate
                v = state["v"] = beta_2*state.get("v", torch.zeros_like(p)) + (1-beta_2)*(grad**2)
                # Compute adjusted alpha for iteration t
                alpha_t = alpha * math.sqrt(1-beta_2**t) / (1 - beta_1**t)
                # Update parameter
                p.data -= alpha_t * m / (torch.sqrt(v) + eps)
                # Apply weight decay
                p.data -= alpha * lamb * p.data
        return loss
    
    def update_lr(self, new_lr: float) -> None:
        for group in self.param_groups:
            group["lr"] = new_lr


def cosine_lr_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    assert cosine_cycle_iters > warmup_iters
    if it<warmup_iters:
        return (it/warmup_iters) * max_learning_rate
    elif it>cosine_cycle_iters:
        return min_learning_rate
    else: # warmup_iters <= it <= cosine_cycle_iters
        anneal_portion = 0.5*(1+math.cos(math.pi * (it - warmup_iters)/(cosine_cycle_iters - warmup_iters)))
        return min_learning_rate + anneal_portion * (max_learning_rate - min_learning_rate)


def clip_gradient(
    parameters: Iterable[torch.nn.Parameter], 
    max_l2_norm: float,
    eps: float = 1e-6 # for numerical stability
) -> None:
    flattened_params = torch.concat([param.grad.flatten() for param in parameters if param.grad is not None])
    l2_norm = torch.sqrt(flattened_params.square().sum())
    if l2_norm >= max_l2_norm:
        for param in parameters:
            if param.grad is not None:
                param.grad *= max_l2_norm/(l2_norm + eps)
    
def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    **kwargs # auxiliary information
) -> None: 
    dict_to_save = {
        "model":model.state_dict(),
        "optimizer":optimizer.state_dict(),
        "iteration":iteration
    }
    dict_to_save.update(kwargs)
    torch.save(dict_to_save, out)

def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    auxiliary_info: dict | None = None # additional info goes into this dictionary
) -> int:
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if auxiliary_info is not None:
        auxiliary_info.update({k:v for k,v in checkpoint.items() if k not in {"model","optimizer","iteration"}})
    return checkpoint["iteration"]

def top_k_sampling(
    k: int
) -> Callable[[Float[torch.Tensor, "batch vocab_size"], torch.Generator], Int[torch.Tensor, " batch "]]:
    """
        Idea of top-k sampling: only sample from the top k vocab in terms of probability.
    """
    def sample(probs: Float[torch.Tensor, "batch vocab_size"], generator: torch.Generator) -> Int[torch.Tensor, " batch "]:
        sorted_idx = torch.argsort(probs, dim=-1)
        # since argsort in ascending order, the last k indices are what we want to sample from.
        # So omit the last k indices, and zero out everything else before sampling
        indices_to_zero_out = sorted_idx[..., :-k]
        chosen_candidates = torch.scatter(probs, dim=-1, index=indices_to_zero_out, value=.0)
        new_normalized_proba = chosen_candidates / chosen_candidates.sum(dim=-1, keepdim=True)
        return torch.multinomial(new_normalized_proba, 1, generator=generator).squeeze(1)


def nucleus_sampling(
    p: int
) -> Callable[[Float[torch.Tensor, "... vocab_size"], torch.Generator], Int[torch.Tensor, " batch "]]:
    """
        Nucleus, or top-p sampling: Only want to sample from the (smallest possible) subset whose cumulative sum of probability is larger than p.
        e.g., if we have a vocab size of 5, and p=0.8
        probs: [0.5, 0.3, 0.15, 0.03, 0.02]
        we want to only sample from the first 2 vocab, as the first 2 vocab forms the smallest subset possible whose cumulative sum is larger or equals to 0.8.

        To do this in a batch -- probs with dimension batch x vocab_size, the trick is to perform masking, and to do it from the other way around, i.e., sort probs in ascending order, and deselect largest subset possible whose cumulative sum is smaller or equals to 1-p
    """
    assert 0 <= p <= 1    
    def sample(probs: Float[torch.Tensor, "batch vocab_size"], generator: torch.Generator) -> Int[torch.Tensor, " batch "]:
        # each row of sorted_idx shows index of ascending order probs for that row
        sorted_idx = torch.argsort(probs, dim=-1)
        # sort each row of probs by ascending order
        sorted_probs = torch.take_along_dim(probs, dim=-1, indices=sorted_idx)
        # take cumulative sum of each row
        sorted_probs_cumsum = torch.cumsum(sorted_probs, dim=-1)
        # assign cumulative sum back to the original unsorted probs
        probs_cumsum = torch.scatter(probs, dim=-1, index=sorted_idx, src=sorted_probs_cumsum)
        # only keep those whose cumsum > 1-p
        # set to 0 so that we don't have negative values, so no need to do a full softmax; a simple division will do
        probs[probs_cumsum <= (1-p)] = 0
        # renormalize remaining vocab that we want to sample from
        new_normalized_proba = probs / probs.sum(dim=-1, keepdim=True)
        return torch.multinomial(new_normalized_proba, 1, generator=generator).squeeze(1)
    return sample


        

def generate_text(
    prompt: str,
    tokenizer: Tokenizer, 
    model: TransformerLM,
    generator: torch.Generator,
    softmax_temperature: float = 1.0,
    sampling_fn: Callable[[Float[torch.Tensor, "... vocab_size"], torch.Generator], Int[torch.Tensor, " batch "]] = nucleus_sampling(0.9),
    max_num_tokens: int = 250, 
    device: torch.device | None = None,
) -> str:
    """
    Given a prompt, convert prompt into tokens, and feed into LM. Autoregressively sample the distribution over vocab for the predicted next word and feed in the most probable word back into the input.
    Keep generating until we receive <|endoftext|> or reach max_num_tokens.
    """
    assert max_num_tokens > 0
    context = tokenizer.encode(prompt)

    termination_tokenIDs = tokenizer.encode("<|endoftext|>")
    assert len(termination_tokenIDs)==1, "special token must be added to vocab as a unique word"
    termination_token = termination_tokenIDs[0]
    sampled_tokens = 0  
    # conditions: keep sampling if not yet reached termination token, or not yet reached max_num_tokens(if exist), or 
    for i in tqdm(range(max_num_tokens), desc="Generating Tokens"):
        if context[-1]==termination_token:
            tqdm.write(f"Reached termination at token {i}; breaking the loop...")
            break
        # model output dim: batch_size x seq_len x vocab_size
        # we only take the last word of the 1st batch
        possible_next_tokens: Float[torch.Tensor, "1 vocab_size"] = model(torch.tensor(context[-model.context_length:], device=device).unsqueeze(dim=0))[0,-1, None] 
        probs = softmax(possible_next_tokens, dim=-1, temperature=softmax_temperature)
        next_tokenID = sampling_fn(probs, generator)[0].item()
        context.append(next_tokenID)
        sampled_tokens += 1

    return tokenizer.decode(context[-sampled_tokens:])
        


    
