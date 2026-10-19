import os
from pathlib import Path
from statistics import mean

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from lib.data import batch_tokenize


def train(
    model: nn.Module,
    optim: Optimizer,
    train_dl: DataLoader,
    val_dl: DataLoader,
    epochs: int,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    single_batch: bool = False,
    save_path: Path | None = None
) -> tuple[list[float], list[float], list[str]]:
    train_losses: list[float] = []
    val_losses: list[float] = []

    if not single_batch:
        generate(model, seq_len=128, batch_size=4, device=device, print_seqs=True)

    seqs: list[str] = []
    for e in range(epochs):
        print(f"Epoch {e + 1}/{epochs}")
        train_losses.append(train_or_val_single_epoch(model, train_dl, optim=optim, device=device, single_batch=single_batch))
        val_losses.append(train_or_val_single_epoch(model, val_dl, device=device, single_batch=single_batch))

        if not single_batch:
            batch_seqs = generate(model, seq_len=128, batch_size=4, device=device, print_seqs=True)
            seqs.append(batch_seqs[0])

        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            torch.save(model.state_dict(), save_path/"checkpoint.pth")
            if val_losses[-1] == min(val_losses):
                torch.save(model.state_dict(), save_path/"best.pth")


    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        torch.save(model.state_dict(), save_path/"full_training.pth")

    return train_losses, val_losses, seqs


def train_or_val_single_epoch(
    model: nn.Module,
    dl: DataLoader,
    optim: Optimizer | None = None,
    device: str = "cpu",
    single_batch: bool = False,
) -> float:
    accs: list[float] = []
    losses: list[float] = []
    loss = nn.CrossEntropyLoss()
    with torch.set_grad_enabled(optim is not None):
        for inp, target in dl:
            if optim is not None:
                optim.zero_grad()

            # Forward
            inp, target = inp.to(device), target.to(device)
            with torch.autocast(dtype=torch.bfloat16, device_type="cuda"):
                out = model(inp)  # BSC
                out = out.permute(0, 2, 1)  # BCS (cross entropy loss requires C to be at dim=1)
                loss_value = torch.mean(loss(out, target))

            # Backward
            if optim is not None:
                loss_value.backward()
                optim.step()

            acc_value = torch.mean(torch.argmax(out, dim=1) == target, dtype=torch.float32)
            accs.append(acc_value.to("cpu").item())
            losses.append(loss_value.to("cpu").item())

            if single_batch:
                break

    print(f"  {'val  ' if optim is None else 'train'}: acc={100 * mean(accs):.2f}%, loss={mean(losses):.3f}")
    return mean(losses)


def generate(
    model: nn.Module,
    seq_len: int,
    batch_size: int,
    prefix="",
    device: str = "cpu",
    strip: bool = True,
    print_seqs: bool = False
) -> list[str]:
    seqs = ["\0" + prefix for _ in range(batch_size)]
    for s in range(len(prefix), seq_len - 1):
        padded_seqs = [seq + "\1" * (seq_len - len(seq)) for seq in seqs]
        tokenized = batch_tokenize(padded_seqs).to(device)
        with torch.autocast(dtype=torch.bfloat16, device_type="cuda"):
            out = model(tokenized)  # BSC
            out = F.softmax(out, dim=2)  # BSC
            out = out.view(batch_size * len(padded_seqs[0]), -1)  # (BS)C
        indices = torch.multinomial(out, 1)  # (BS)1
        indices = indices.view(batch_size, len(padded_seqs[0])).to("cpu")  # BS
        next_idxs = [indices[b][s] for b in range(batch_size)]
        seqs = [seq + chr(idx) for seq, idx in zip(seqs, next_idxs, strict=True)]

    if strip:
        seqs = [s.strip("\0").strip("\1") for s in seqs]
    if print_seqs:
        for s in seqs:
            print(repr(s))
    return seqs
