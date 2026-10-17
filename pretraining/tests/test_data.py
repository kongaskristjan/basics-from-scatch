from lib.data import JokesDataset, get_jokes


def test_get_jokes():
    jokes = get_jokes()
    assert len(jokes) > 0 and len(jokes[0]) > 0


def test_dataset():
    # Test dataset size properties
    len1 = 16
    train, val = JokesDataset.create_train_and_val(len1, full=False)
    assert len(train) > len(val)
    sample = train[1]

    # Test next token prediction
    assert sample[0][2] == sample[1][1]
    assert sample[0][3] == sample[1][2]

    # Test that not all tokens are equal
    equal = True
    for i in range(len1):
        if sample[0][i] != sample[0][0] or sample[1][i] != sample[1][0]:
            equal = False
    assert not equal

    # Test truncation
    assert len(sample[0]) == len(sample[1]) == len1

    # Test padding
    len2 = 4096
    train, val = JokesDataset.create_train_and_val(len2, full=False)
    sample = train[1]
    assert len(sample[0]) == len(sample[1]) == len2


def test_dataset_characters():
    train, _val = JokesDataset.create_train_and_val(10, full=False)
    inp, target = train.get_strings(0)
    assert inp[0] == "\0"
    assert target[0] != "\0" and target[-1] == "\1"
    assert inp[1:] == target[:-1]
