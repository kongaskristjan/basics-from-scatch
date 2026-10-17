from pathlib import Path

import torch

from lib.data import get_jokes_data_loaders
from lib.model import Transformer
from lib.training import train


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dl, val_dl = get_jokes_data_loaders(str_length=64, num_workers=0)
    model = Transformer(num_blocks=4, heads=4, channels=64).to(device)
    optim = torch.optim.AdamW(model.parameters())
    _train_losses, _val_losses = train(model, optim, train_dl, val_dl, epochs=50, device=device, save_path=Path("models/"))


if __name__ == "__main__":
    main()
