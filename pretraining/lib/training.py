from statistics import mean

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


def train(
    model: nn.Module,
    optim: Optimizer,
    train_dl: DataLoader,
    val_dl: DataLoader,
    epochs: int,
    device: str = "cpu",
    single_batch: bool = False,
) -> tuple[list[float], list[float]]:
    train_losses: list[float] = []
    val_losses: list[float] = []
    for e in range(epochs):
        print(f"Epoch {e + 1}/{epochs}")
        train_losses.append(train_or_val_single_epoch(model, train_dl, optim=optim, device=device, single_batch=single_batch))
        val_losses.append(train_or_val_single_epoch(model, val_dl, device=device, single_batch=single_batch))
    return train_losses, val_losses


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

            inp = inp.to(device)
            target = target.to(device)
            out = model(inp)  # BSC
            out = out.permute(0, 2, 1)  # BCS (cross entropy loss requires C to be at dim=1)
            loss_value = torch.mean(loss(out, target))
            if optim is not None:
                loss_value.backward()
                optim.step()

            acc_value = torch.mean(torch.argmax(out, dim=1) == target, dtype=torch.float32)
            accs.append(acc_value.to("cpu").item())
            losses.append(loss_value.to("cpu").item())

            if single_batch:
                break

    print(f"  {'val  ' if optim is None else 'train'}: acc={100 * mean(accs):.1f}%, loss={mean(losses):.1f}")
    return mean(losses)
