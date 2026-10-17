import random

import torch
from cached_path import cached_path
from torch.utils.data import Dataset


class JokesDataset(Dataset):
    @staticmethod
    def create_train_and_val(
        str_length: int = 128,
        val_pct: float = 0.2,
        seed: int = 42,
        full: bool = True
    ) -> tuple[JokesDataset, JokesDataset]:
        rng = random.Random(seed)

        jokes = get_jokes(full=full)
        rng.shuffle(jokes)
        cutoff = int(val_pct * len(jokes))
        val = JokesDataset(str_length, jokes[:cutoff])
        train = JokesDataset(str_length, jokes[cutoff:])
        return train, val

    def __init__(self, str_length: int, jokes: list[str]):
        super().__init__()
        self.jokes = jokes
        for i, s in enumerate(self.jokes):
            trun = "\0" + s[:str_length - 1]
            padded = trun + (str_length - len(trun)) * "\1"
            self.jokes[i] = padded

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        inp, target = self.get_strings(index)
        inp_tok = tokenize(inp)
        target_tok = tokenize(target)
        return inp_tok, target_tok

    def __len__(self) -> int:
        return len(self.jokes)

    def get_strings(self, index: int) -> tuple[str, str]:
        inp = self.jokes[index]
        target = inp[1:] + "\1"
        return inp, target


def tokenize(string: str) -> torch.Tensor:
    tokens = []
    for chr in string:
        value = ord(chr)
        value = min(value, 255)
        tokens.append(value)
    tokens_t = torch.LongTensor(tokens)
    return tokens_t


def batch_tokenize(strs: list[str]) -> torch.Tensor:
    assert len(strs) > 0
    len_str = len(strs[0])
    assert all(len_str == len(s) for s in strs)

    tokenized = [tokenize(s) for s in strs]
    tensor = torch.stack(tokenized)
    return tensor


def get_jokes(full: bool = True) -> list[str]:
    jokes_path = cached_path("https://github.com/Fraser-Greenlee/my-huggingface-datasets/raw/refs/heads/master/data/short-jokes/train.txt")
    with open(jokes_path) as f:
        if full:
            jokes_string = f.read()
        else:
            jokes_string = f.read(5000)
    jokes = jokes_string.split("\n")
    return jokes
