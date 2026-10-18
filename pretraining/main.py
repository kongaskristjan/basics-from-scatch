from pathlib import Path

import torch

from lib.data import get_jokes_data_loaders
from lib.model import Transformer
from lib.training import train


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dl, val_dl = get_jokes_data_loaders(str_length=64, num_workers=0)
    model = Transformer(num_blocks=6, heads=4, channels=1024).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=3e-5)
    _train_losses, _val_losses, seqs = train(model, optim, train_dl, val_dl, epochs=100, device=device, save_path=Path("models/"))
    save_seqs(seqs, Path("models")/"seqs.txt")


def save_seqs(seqs: list[str], output_path: Path) -> None:
    with open(output_path, "w") as f:
        f.write("\n".join(seqs))


if __name__ == "__main__":
    main()
