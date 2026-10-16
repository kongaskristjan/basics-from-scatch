# ty: ignore[invalid-mutable-override]

import math

import torch
from torch import nn
from torch.nn import functional as F

class Transformer(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        heads: int,
        channels: int,
        dict_size: int,
    ):
        super().__init__()
        assert channels % heads == 0

        self.in_embedding = nn.Embedding(dict_size, channels)
        self.mlps: list[MLP] = []
        self.attns: list[MultiHeadAttention] = []
        self.norm = RMSNorm()
        for i in range(num_blocks):
            self.mlps.append(MLP(channels))
            self.attns.append(MultiHeadAttention(heads, channels))
        self.out_embedding = nn.Linear(channels, dict_size)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        # inp: (batch_size, sequence_length), int64
        # skip: (batch_size, sequence_length, channels), float
        skip = self.in_embedding(inp)
        for mlp, attn in zip(self.mlps, self.attns, strict=True):
            x = mlp(skip)
            skip = self.norm(x)
            x = attn(x)
            skip = self.norm(x)
        out = self.out_embedding(skip)
        return out


class MLP(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.layer1 = nn.Linear(channels, channels)
        self.layer2 = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, channels = x.shape
        x = x.view(batch_size * sequence_length, channels)
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = x.view(batch_size, sequence_length, channels)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, heads: int, channels: int):
        super().__init__()

        self.heads = heads
        self.w_key = nn.Linear(channels, channels)
        self.w_query = nn.Linear(channels, channels)
        self.w_value = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, channels = x.shape
        assert channels % self.heads == 0
        seq_batch, heads, attn_channels = batch_size * sequence_length, self.heads, channels // self.heads

        x = x.view(seq_batch, channels)
        key = self.w_key(x).view(seq_batch * heads, attn_channels)
        query = self.w_query(x).view(seq_batch * heads, attn_channels)
        value = self.w_value(x).view(seq_batch * heads, attn_channels)
        attention = torch.inner(key, query) / math.sqrt(attn_channels)
        





class RMSNorm(nn.Module):
    pass
