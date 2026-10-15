import random

from cached_path import cached_path
from torch.utils.data import Dataset


class JokesDataset(Dataset):
    @staticmethod
    def create_train_and_val(str_length: int = 128, val_pct: float = 0.2, seed: int = 42) -> tuple[JokesDataset, JokesDataset]:
        rng = random.Random(seed)

        jokes = get_jokes()
        rng.shuffle(jokes)
        cutoff = int(val_pct * len(jokes))
        val = JokesDataset(str_length, jokes[:cutoff])
        train = JokesDataset(str_length, jokes[cutoff:])
        return train, val

    def __init__(self, str_length: int, jokes: list[str]):
        super().__init__()
        self.jokes = jokes
        for i, s in enumerate(self.jokes):
            trun = s[:str_length]
            padded = trun + (str_length - len(trun)) * "\0"
            self.jokes[i] = padded

    def __getitem__(self, index: int) -> tuple[str, str]:
        inp = self.jokes[index]
        target = inp[1:] + "\0"
        return inp, target

    def __len__(self) -> int:
        return len(self.jokes)


def get_jokes() -> list[str]:
    jokes_path = cached_path("https://github.com/Fraser-Greenlee/my-huggingface-datasets/raw/refs/heads/master/data/short-jokes/train.txt")
    with open(jokes_path) as f:
        jokes_string = f.read()
    jokes = jokes_string.split("\n")
    return jokes
