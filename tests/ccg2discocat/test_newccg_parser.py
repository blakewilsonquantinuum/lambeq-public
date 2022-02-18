import pytest

from lambeq import NewCCGParser


@pytest.fixture(scope='module')
def newccg_parser():
    return NewCCGParser()


def test_sentence2diagram(newccg_parser):
    sentence = 'What Alice is and is not .'
    assert newccg_parser.sentence2diagram(sentence) is not None
