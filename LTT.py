import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from tst.encoder import Encoder
from tst.decoder import Decoder
from tst.utils import generate_original_PE, generate_regular_PE



class temporal_transformer(nn.Module):

    def __init__(self,
                 d_input: int,
                 d_model: int,
                 d_output: int,
                 q: int,
                 v: int,
                 h: int,
                 N: int,
                 attention_size: int = None,
                 dropout: float = 0.3,
                 chunk_mode: str = 'window',
                 pe: str = None,
                 # pe_period: str = None
                 pe_period: int = 6
                 ):
        """Create transformer structure from Encoder and Decoder blocks."""
        super().__init__()

        self._d_model = d_model

        self.encoder1 = Encoder(d_model,q,v,h,attention_size=attention_size,
                                            dropout=dropout,
                                            chunk_mode=chunk_mode)
        self.encoder2 = Encoder(d_model, q, v, h, attention_size=attention_size,
                                dropout=dropout,
                                chunk_mode=chunk_mode)
        self.encoder3 = Encoder(d_model, q, v, h, attention_size=attention_size,
                                dropout=dropout,
                                chunk_mode=chunk_mode)
        self.decoder1 = Decoder(d_model, q, v, h, attention_size=attention_size,
                                dropout=dropout,
                                chunk_mode=chunk_mode)
        self.decoder2 = Decoder(d_model, q, v, h, attention_size=attention_size,
                                dropout=dropout,
                                chunk_mode=chunk_mode)
        self.decoder3 = Decoder(d_model, q, v, h, attention_size=attention_size,
                                dropout=dropout,
                                chunk_mode=chunk_mode)
        self.layers_encoding = nn.ModuleList([Encoder(d_model,
                                                      q,
                                                      v,
                                                      h,
                                                      attention_size=attention_size,
                                                      dropout=dropout,
                                                      chunk_mode=chunk_mode) for _ in range(N)])
        self.layers_decoding = nn.ModuleList([Decoder(d_model,
                                                      q,
                                                      v,
                                                      h,
                                                      attention_size=attention_size,
                                                      dropout=dropout,
                                                      chunk_mode=chunk_mode) for _ in range(N)])

        self._embedding = nn.Linear(d_input, d_model)
        self._linear = nn.Linear(d_model, d_output)
        # self.final_onv = nn.Conv2d(325, 325, (12, 1), 1)
        self.final_onv = nn.Conv2d(228, 228, (12, 1), 1)
        pe_functions = {
            'original': generate_original_PE,
            'regular': generate_regular_PE,
        }

        if pe in pe_functions.keys():
            self._generate_PE = pe_functions[pe]
            self._pe_period = pe_period
        elif pe is None:
            self._generate_PE = None
        else:
            raise NameError(
                f'PE "{pe}" not understood. Must be one of {", ".join(pe_functions.keys())} or None.')

        self.name = 'transformer'

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        #x = x.permute(0, 3, 2, 1)  #only for stgcnt
        K = x.shape[2]

        # Embeddin module
        encoding = self._embedding(x)

        # Add position encoding
        if self._generate_PE is not None:
            pe_params = {'period': self._pe_period} if self._pe_period else {}
            positional_encoding = self._generate_PE(K, self._d_model, **pe_params)
            positional_encoding = positional_encoding.to(encoding.device)
            encoding.add_(positional_encoding)

        # Encoding stack
        x1 = self.encoder1(encoding)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)

        # Decoding stack
        # decoding = encoding
        decoding = x3

        # Add position encoding
        if self._generate_PE is not None:
            positional_encoding = self._generate_PE(K, self._d_model)
            positional_encoding = positional_encoding.to(decoding.device)
            decoding.add_(positional_encoding)

        x4 = self.decoder1(decoding,x3)
        x5 = self.decoder2(x4,x3)
        x6 = self.decoder3(x5,x3)
        x6 = self._linear(x6)

        # Output module

        output = self._linear(decoding)
        output = self.final_onv(output)


        return output, x1, x2, x6

