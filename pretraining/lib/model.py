# ty: ignore[invalid-mutable-override]

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
        self.norm = nn.RMSNorm([channels,])
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
        # Constants
        dtype = x.dtype
        device = x.device

        batch_size, seq_len, channels = x.shape  # BI(HC)
        assert channels % self.heads == 0
        heads, attn_channels = self.heads, channels // self.heads

        # Causal mask
        # attn_mask will be zeros for [?, 0, 0..seq_len]
        # thus dimensions: [B, input, output]
        attn_mask = torch.full((seq_len, seq_len), float("-inf"), dtype=dtype, device=device)
        attn_mask = torch.triu(attn_mask, diagonal = 1)
        attn_mask = attn_mask.view(1, seq_len, seq_len)  # 1OI
        attn_mask = attn_mask.permute(0, 2, 1)  # 1IO

        x = x.view(batch_size * seq_len, channels)  # (BI)(HC)

        # Key and query vectors
        key = self.w_key(x)
        key = key.view(batch_size, seq_len, heads, attn_channels)  # BIHC
        key = key.permute(0, 2, 1, 3)  # BHIC
        assert key.shape == (batch_size, heads, seq_len, attn_channels)
        key = key.reshape(batch_size * heads, seq_len, attn_channels)  # (BH)IC

        query = self.w_query(x)  # (BO)(HC)
        query = query.view(batch_size, seq_len, heads, attn_channels)  # BOHC
        query = query.permute(0, 2, 3, 1)  # BHCO
        assert query.shape == (batch_size, heads, attn_channels, seq_len)
        query = query.reshape(batch_size * heads, attn_channels, seq_len)  # (BH)CO

        # Attention
        scores = torch.bmm(key, query)  # (BH)IO
        assert scores.shape == (batch_size * heads, seq_len, seq_len)
        causal_scores = scores + attn_mask  # (BH)IO
        attn = torch.softmax(causal_scores, dim=1)
        attn = attn.view(batch_size * heads, seq_len, seq_len)  # BHIO
        attn = attn.permute(0, 2, 1)  # (BH)OI
        assert attn.shape == (batch_size * heads, seq_len, seq_len)

        # Value vector
        value = self.w_value(x)
        value = value.view(batch_size, seq_len, heads, attn_channels)  # BIHC
        value = value.permute(0, 2, 1, 3)  # BHIC
        assert value.shape == (batch_size, heads, seq_len, attn_channels)
        value = value.reshape(batch_size * heads, seq_len, attn_channels)  # (BH)IC

        # Attn result
        result = torch.bmm(attn, value)  # (BH)OC
        assert result.shape == (batch_size * heads, seq_len, attn_channels)
        result = result.view(batch_size, heads, seq_len, attn_channels)  # BHOC
        result = result.permute(0, 2, 1, 3)  # BOHC
        result = result.reshape(batch_size, seq_len, channels)  # BO(HC)

        return result
