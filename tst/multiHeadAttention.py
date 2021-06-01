from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tst.utils import generate_local_map_mask


class MultiHeadAttention(nn.Module):

    def __init__(self,
                 d_model: int,
                 q: int,
                 v: int,
                 h: int,
                 attention_size: int = None):
        """Initialize the Multi Head Block."""
        super().__init__()

        self._h = h
        self._attention_size = attention_size

        # Query, keys and value matrices
        self._W_q = nn.Linear(d_model, q*self._h)
        self._W_k = nn.Linear(d_model, q*self._h)
        self._W_v = nn.Linear(d_model, v*self._h)

        # Output linear function
        self._W_o = nn.Linear(self._h*v, d_model)

        # Score placeholder
        self._scores = None

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[str] = None) -> torch.Tensor:

        K = query.shape[2]

        # Compute Q, K and V, concatenate heads on batch dimension
        queries = torch.cat(self._W_q(query).chunk(self._h, dim=-1), dim=0)
        keys = torch.cat(self._W_k(key).chunk(self._h, dim=-1), dim=0)
        values = torch.cat(self._W_v(value).chunk(self._h, dim=-1), dim=0)

        # Scaled Dot Product
        self._scores = torch.matmul(queries, keys.transpose(2, 3)) / np.sqrt(K)

        # Compute local map mask
        if self._attention_size is not None:
            attention_mask = generate_local_map_mask(K, self._attention_size, mask_future=False, device=self._scores.device)
            self._scores = self._scores.masked_fill(attention_mask, float('-inf'))

        # Compute future mask
        if mask == "subsequent":
            future_mask = torch.triu(torch.ones((K, K)), diagonal=1).bool()
            future_mask = future_mask.to(self._scores.device)
            self._scores = self._scores.masked_fill(future_mask, float('-inf'))

        # Apply sotfmax
        self._scores = F.softmax(self._scores, dim=-1)

        attention = torch.matmul(self._scores, values)

        # Concatenat the heads
        attention_heads = torch.cat(attention.chunk(self._h, dim=0), dim=-1)

        # Apply linear transformation W^O
        self_attention = self._W_o(attention_heads)

        return self_attention

    @property
    def attention_map(self) -> torch.Tensor:

        if self._scores is None:
            raise RuntimeError(
                "Evaluate the model once to generate attention map")
        return self._scores


class MultiHeadAttentionChunk(MultiHeadAttention):

    def __init__(self,
                 d_model: int,
                 q: int,
                 v: int,
                 h: int,
                 attention_size: int = None,
                 chunk_size: Optional[int] = 168,
                 **kwargs):
        """Initialize the Multi Head Block."""
        super().__init__(d_model, q, v, h, attention_size, **kwargs)

        self._chunk_size = chunk_size

        # Score mask for decoder
        self._future_mask = nn.Parameter(torch.triu(torch.ones((self._chunk_size, self._chunk_size)), diagonal=1).bool(),
                                         requires_grad=False)

        if self._attention_size is not None:
            self._attention_mask = nn.Parameter(generate_local_map_mask(self._chunk_size, self._attention_size),
                                                requires_grad=False)

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[str] = None) -> torch.Tensor:

        K = query.shape[1]
        n_chunk = K // self._chunk_size

        # Compute Q, K and V, concatenate heads on batch dimension
        queries = torch.cat(torch.cat(self._W_q(query).chunk(self._h, dim=-1), dim=0).chunk(n_chunk, dim=1), dim=0)
        keys = torch.cat(torch.cat(self._W_k(key).chunk(self._h, dim=-1), dim=0).chunk(n_chunk, dim=1), dim=0)
        values = torch.cat(torch.cat(self._W_v(value).chunk(self._h, dim=-1), dim=0).chunk(n_chunk, dim=1), dim=0)

        # Scaled Dot Product
        self._scores = torch.bmm(queries, keys.transpose(1, 2)) / np.sqrt(self._chunk_size)

        # Compute local map mask
        if self._attention_size is not None:
            self._scores = self._scores.masked_fill(self._attention_mask, float('-inf'))

        # Compute future mask
        if mask == "subsequent":
            self._scores = self._scores.masked_fill(self._future_mask, float('-inf'))

        # Apply softmax
        self._scores = F.softmax(self._scores, dim=-1)

        attention = torch.bmm(self._scores, values)

        # Concatenat the heads
        attention_heads = torch.cat(torch.cat(attention.chunk(
            n_chunk, dim=0), dim=1).chunk(self._h, dim=0), dim=-1)

        # Apply linear transformation W^O
        self_attention = self._W_o(attention_heads)

        return self_attention


class MultiHeadAttentionWindow(MultiHeadAttention):


    def __init__(self,
                 d_model: int,
                 q: int,
                 v: int,
                 h: int,
                 attention_size: int = None,
                 window_size: Optional[int] = 12,
                 # padding: Optional[int] = 168 // 4,
                 **kwargs):
        """Initialize the Multi Head Block."""
        super().__init__(d_model, q, v, h, attention_size, **kwargs)

        self._window_size = window_size
        # self._padding = padding
        self._q = q
        self._v = v

        # Step size for the moving window
        self._step = self._window_size

        # Score mask for decoder
        self._future_mask = nn.Parameter(torch.triu(torch.ones((self._window_size, self._window_size)), diagonal=1).bool(),
                                         requires_grad=False)

        if self._attention_size is not None:
            self._attention_mask = nn.Parameter(generate_local_map_mask(self._window_size, self._attention_size),
                                                requires_grad=False)

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[str] = None) -> torch.Tensor:

        batch_size = query.shape[0]

        # Apply padding to input sequence
        # query = F.pad(query.transpose(1, 2), (self._padding, self._padding), 'replicate').transpose(1, 2)
        # key = F.pad(key.transpose(1, 2), (self._padding, self._padding), 'replicate').transpose(1, 2)
        # value = F.pad(value.transpose(1, 2), (self._padding, self._padding), 'replicate').transpose(1, 2)

        # Compute Q, K and V, concatenate heads on batch dimension
        queries = torch.cat(self._W_q(query).chunk(self._h, dim=-1), dim=0)
        keys = torch.cat(self._W_k(key).chunk(self._h, dim=-1), dim=0)
        values = torch.cat(self._W_v(value).chunk(self._h, dim=-1), dim=0)

        # Divide Q, K and V using a moving window
        queries = queries.unfold(dimension=1, size=self._window_size, step=self._step).reshape((-1, self._q, self._window_size)).transpose(1, 2)
        keys = keys.unfold(dimension=1, size=self._window_size, step=self._step).reshape((-1, self._q, self._window_size)).transpose(1, 2)
        values = values.unfold(dimension=1, size=self._window_size, step=self._step).reshape((-1, self._v, self._window_size)).transpose(1, 2)

        # Scaled Dot Product
        self._scores = torch.bmm(queries, keys.transpose(1, 2)) / np.sqrt(self._window_size)

        # Compute local map mask
        if self._attention_size is not None:
            self._scores = self._scores.masked_fill(self._attention_mask, float('-inf'))

        # Compute future mask
        if mask == "subsequent":
            self._scores = self._scores.masked_fill(self._future_mask, float('-inf'))

        # Apply softmax
        self._scores = F.softmax(self._scores, dim=-1)

        attention = torch.bmm(self._scores, values)

        # Fold chunks back
        attention = attention.reshape((batch_size*self._h, -1, self._window_size, self._v))

        # Concatenat the heads
        attention_heads = torch.cat(attention.chunk(self._h, dim=0), dim=-1)

        # Apply linear transformation W^O
        self_attention = self._W_o(attention_heads)

        return self_attention
