
from lib.data import get_jokes


def main() -> None:
    jokes = get_jokes()
    for i in range(min(10, len(jokes))):
        print(jokes[i])


if __name__ == "__main__":
    main()
