import math

import torch
from torch import nn
import einx

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

    def forward(self, x:torch.Tensor) -> torch.Tensor:
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

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return einx.get_at("[vocab] d_model, batch_size (seq_len [i]) -> batch_size seq_len d_model", self.embed_mat, token_ids, i=1)
