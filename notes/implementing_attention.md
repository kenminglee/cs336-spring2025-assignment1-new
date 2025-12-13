Recall the Attention operation:

$$\text{Attention}(Q,K,V)=
\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

where:
- $Q$ has dimension (batch-size, $\ldots$, seq-len-q, $d_k$)
- $K$ has dimension (batch-size, $\ldots$, seq-len-k, $d_k$)
- $V$ has dimension (batch-size, $\ldots$, seq-len-k, $d_v$)

In the case of masked attention, we are also given a mask of dimension (batch-size, $\ldots$, seq-len-q, seq-len-k).

**Intuition of** $\mathbf{QK^T}$ 

After the $QK^T$ operation, we obtain a tensor with dimension (batch-size, $\ldots$, seq-len-q, seq-len-k).

Intuitively, assuming a batch-size of 1, $QK^T$ results in a 2D matrix of dimension (seq-len-q, seq-len-k).
In this case, each element $x_{ij}$ represents the amount of attention that the $i$-th token of q should pay to $j$-th token of k.
Note that q and k are not necessarily the same, such as in the case of cross-attention, where q is computed from the decoder while k is computed from the encoder (think of cases like translation).

**Intuition of Masking**

Additionally, in the case of training causal decoders (e.g., GPTs), we want to ensure that the future tokens are masked, ensuring that the current token output is only conditioned on the past tokens.
This is performed by masking, which is to set attention output of $QK^T$ to -$\infty$ for future tokens (i.e., $x_{ij} \rightarrow -\infty, \, \forall j>i$).

Since $QK^T$ results in a tensor of dimension (batch-size, $\ldots$, seq-len-q, seq-len-k), the mask also needs to have the same dimension.

**Which dimension to take softmax over?**

Recall that $QK^T$ results in a tensor of dimension (batch-size, $\ldots$, seq-len-q, seq-len-k).

Given that we want to know how much token $i$ in $Q$ (i.e., token in position $i$) should attend to each token $j$ in $K$, this means that we should normalize over all tokens in $K$ for each token $i$, hence we take the softmax over the final axis.

**Intuition of multiplying by V**

After computing the softmax, multiplying with $V$ is akin to taking a weighted average of the value of each token (in $K$).
