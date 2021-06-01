import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tst.multiHeadAttention import MultiHeadAttention, MultiHeadAttentionChunk, MultiHeadAttentionWindow
from tst.positionwiseFeedForward import PositionwiseFeedForward


class Decoder(nn.Module):

    def __init__(self,
                 d_model: int,
                 q: int,
                 v: int,
                 h: int,
                 attention_size: int = None,
                 dropout: float = 0.3,
                 chunk_mode: str = 'chunk'):
        super().__init__()

        chunk_mode_modules = {
            'chunk': MultiHeadAttentionChunk,
            'window': MultiHeadAttentionWindow,
        }

        if chunk_mode in chunk_mode_modules.keys():
            MHA = chunk_mode_modules[chunk_mode]
        elif chunk_mode is None:
            MHA = MultiHeadAttention
        else:
            raise NameError(
                f'chunk_mode "{chunk_mode}" not understood. Must be one of {", ".join(chunk_mode_modules.keys())} or None.')

        self._selfAttention = MHA(d_model, q, v, h, attention_size=attention_size)
        self._encoderDecoderAttention = MHA(d_model, q, v, h, attention_size=attention_size)
        self._feedForward = PositionwiseFeedForward(d_model)

        self._layerNorm1 = nn.LayerNorm(d_model)
        self._layerNorm2 = nn.LayerNorm(d_model)
        self._layerNorm3 = nn.LayerNorm(d_model)

        self._dopout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:

        # Self attention
        residual = x
        x = self._selfAttention(query=x, key=x, value=x, mask="subsequent")
        x = self._dopout(x)
        x = self._layerNorm1(x + residual)

        # Encoder-decoder attention
        residual = x
        x = self._encoderDecoderAttention(query=x, key=memory, value=memory)
        x = self._dopout(x)
        x = self._layerNorm2(x + residual)

        # Feed forward
        residual = x
        x = self._feedForward(x)
        x = self._dopout(x)
        x = self._layerNorm3(x + residual)

        return x
