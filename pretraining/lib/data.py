from cached_path import cached_path


def get_jokes() -> list[str]:
    jokes_path = cached_path("https://github.com/Fraser-Greenlee/my-huggingface-datasets/raw/refs/heads/master/data/short-jokes/train.txt")
    with open(jokes_path) as f:
        jokes_string = f.read()
    jokes = jokes_string.split("\n")
    return jokes
