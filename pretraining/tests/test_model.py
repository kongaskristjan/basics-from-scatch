import torch
from torch import nn

from lib.data import batch_tokenize, tokenize
from lib.model import MLP, MultiHeadAttention, Transformer
from tests.fixtures import fix_seeds  # noqa: F401


def test_transformer(fix_seeds):  # noqa: F811
    num_blocks, heads, channels, dict_size = 4, 2, 8, 256
    transformer = Transformer(num_blocks, heads, channels, dict_size)

    inp = batch_tokenize([
        "aaaaabbbbb",
        "z./rftewt ",
    ])
    out = transformer(inp)

    assert out.shape == (2, 10, dict_size)
    assert torch.allclose(out[0, 0:5], out[0, :1])
    assert not torch.allclose(out[0, 5:10], out[0, 5:6])
    assert not torch.allclose(out[0], out[0, :1])


def test_mlp(fix_seeds):  # noqa: F811
    channels = 10
    in_embedding = nn.Embedding(256, channels)
    mlp = MLP(channels)

    inp = batch_tokenize([
        "aaaaabbbbb",
        "z./rftewt ",
    ])
    inp_emb = in_embedding(inp)
    out = mlp(inp_emb)

    assert inp_emb.shape == out.shape
    assert torch.allclose(out[0, 0:5], out[0, :1])
    assert torch.allclose(out[0, 5:10], out[0, 5:6])
    assert not torch.allclose(out[0], out[0, :1])


def test_attention_causality(fix_seeds):  # noqa: F811
    heads, channels = 1, 1
    in_embedding = nn.Embedding(256, channels)
    attn = MultiHeadAttention(heads=heads, channels=channels)

    inp = tokenize("aaaabaaaa").unsqueeze(0)
    inp_emb = in_embedding(inp)
    out = attn(inp_emb).squeeze(0)

    # First four a's must have identical value, as we're just taking the weighted average os identical values
    assert torch.allclose(out[0:4], out[0])

    # The next values must differ
    assert torch.all(torch.logical_not(torch.isclose(out[4:9], out[0])))


def test_attention_heads_channels(fix_seeds):  # noqa: F811
    heads, channels = 2, 4
    in_embedding = nn.Embedding(256, channels)
    attn = MultiHeadAttention(heads=heads, channels=channels)

    inp = batch_tokenize([
        "aaaaabbbbb",
        "aaaaabbbbb",  # Identical to 0th element
        "aaaaacbade",
        "z./rftewt ",
    ])
    inp_emb = in_embedding(inp)
    out = attn(inp_emb)

    # Test batch
    assert torch.allclose(out[0], out[1])
    assert torch.allclose(out[0][:5], out[2][:5])
    assert not torch.allclose(out[0], out[2])
    assert not torch.allclose(out[0], out[3])
