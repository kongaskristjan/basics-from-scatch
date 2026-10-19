# ty: ignore[invalid-mutable-override]

import torch
from torch import nn

from lib.data import get_jokes_data_loaders
from lib.training import generate, train
from tests.fixtures import fix_seeds  # noqa: F401


def test_fit_single_sequence(fix_seeds):  # noqa: F811
    model = LinearPerToken(256, 4)
    optim = torch.optim.AdamW(model.parameters())
    train_dl, val_dl = get_jokes_data_loaders(str_length=16, batch_size=16, num_workers=0)
    train_losses, val_losses, _seqs = train(model, optim, train_dl, val_dl, epochs=200, single_batch=True)

    assert train_losses[-1] < train_losses[0] - 0.5
    assert val_losses[-1] < val_losses[0] - 0.5


def test_generate():
    model = LinearPerToken(256, 4)

    seqs = generate(model, 128, 2, strip=False)
    assert len(seqs) == 2
    assert len(seqs[0]) == len(seqs[1]) == 128
    assert seqs[0][0] == seqs[1][0] == "\0"

    seqs = generate(model, 128, 2, strip=False, prefix="Hello.")
    assert len(seqs[0]) == len(seqs[1]) == 128


class LinearPerToken(nn.Module):
    def __init__(self, dict_size: int, hidden: int):
        super().__init__()
        self.emb = nn.Embedding(dict_size, hidden)
        self.linear = nn.Linear(hidden, dict_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length = x.shape
        x.view(batch_size * sequence_length)
        x = self.emb(x)
        x = self.linear(x)
        x.view(batch_size, sequence_length, -1)
        return x
