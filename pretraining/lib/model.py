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
        self.attns: list[MHAttention] = []
        self.norm = RMSNorm()
        for i in range(num_blocks):
            self.mlps.append(MLP(channels))
            self.attns.append(MHAttention(channels))
        self.out_embedding = nn.Linear(channels, dict_size)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        skip = self.in_embedding(inp)
        for mlps, attns in zip(self.mlps, self.attns, strict=True):
            for mlp, attn in zip(mlps, attns, strict=True):
                x = mlp(skip)
                skip = norm(x)
                x = attn(x)
                skip = norm(x)
        out = self.out_embedding(skip)
        return skip


class MLP(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.layer1 = nn.Linear(channels, channels)
        self.layer2 = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        return x