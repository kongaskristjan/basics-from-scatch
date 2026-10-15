from lib.data import JokesDataset, get_jokes


def test_get_jokes():
    jokes = get_jokes()
    assert len(jokes) > 0 and len(jokes[0]) > 0

def test_dataset():
    # Test dataset size properties
    len1 = 16
    train, val = JokesDataset.create_train_and_val(len1)
    assert len(train) > len(val)
    sample = train[1]

    # Test next token prediction
    assert sample[0][2] == sample[1][1]
    assert sample[0][3] == sample[1][2]

    # Test truncation
    assert len(sample[0]) == len(sample[1]) == len1

    # Test padding
    len2 = 4096
    train, val = JokesDataset.create_train_and_val(len2)
    sample = train[1]
    assert len(sample[0]) == len(sample[1]) == len2

