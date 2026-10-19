import argparse
from pathlib import Path

import torch

from lib.data import get_jokes_data_loaders
from lib.model import Transformer
from lib.training import generate, train


def main() -> None:
    default_device = "cpu"
    if torch.cuda.is_available(): default_device = "cuda"
    if torch.mps.is_available(): default_device = "mps"
    default_dtype = "float32"
    if torch.cuda.is_bf16_supported() or default_device == "mps":
        default_dtype = "bfloat16"

    # create the top-level parser
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--device", type=str, default=default_device)
    train_parser.add_argument("--dtype", type=str, default=default_dtype)
    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--device", type=str, default=default_device)
    generate_parser.add_argument("--dtype", type=str, default=default_dtype)
    generate_parser.add_argument("--checkpoint", type=str, required=True)

    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    assert isinstance(dtype, torch.dtype)

    if args.command == "train":
        train_command(args.device, dtype)
    elif args.command == "generate":
        generate_command(args.device, dtype, args.checkpoint)
    else:
        assert False, f"No such command `{args.command}`"


def train_command(device: str, dtype: torch.dtype):
    train_dl, val_dl = get_jokes_data_loaders(str_length=128, num_workers=0)
    model = get_model().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=3e-5)
    _train_losses, _val_losses, seqs = train(model, optim, train_dl, val_dl, epochs=100, device=device, dtype=dtype, save_path=Path("models/"))
    save_seqs(seqs, Path("models")/"seqs.txt")


def generate_command(
    device: str,
    dtype: torch.dtype,
    checkpoint_path: Path,
):
    model = get_model().to(device)
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    model.eval()
    prefix = input("Enter completion prefix: ")
    generate(model, seq_len=256, batch_size=4, prefix=prefix, device=device, dtype=dtype, print_seqs=True)


def get_model() -> Transformer:
    return Transformer(num_blocks=6, heads=4, channels=1024)


def save_seqs(seqs: list[str], output_path: Path) -> None:
    with open(output_path, "w") as f:
        f.write("\n".join(seqs))


if __name__ == "__main__":
    main()
