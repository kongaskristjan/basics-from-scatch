from lib.data import get_jokes


def test_get_jokes():
    assert len(get_jokes()) > 0
