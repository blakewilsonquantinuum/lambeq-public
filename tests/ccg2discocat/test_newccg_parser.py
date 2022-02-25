import pytest

from lambeq import CCGAtomicType, NewCCGParser


@pytest.fixture(scope='module')
def newccg_parser():
    return NewCCGParser()


def test_sentence2diagram(newccg_parser):
    sentence = 'What Alice is and is not .'
    assert newccg_parser.sentence2diagram(sentence) is not None


def test_root_filtering(newccg_parser):
    S = CCGAtomicType.SENTENCE
    N = CCGAtomicType.NOUN

    restricted_parser = NewCCGParser(root_cats=['NP'])

    sentence1 = 'do'
    assert newccg_parser.sentence2tree(sentence1).biclosed_type == N >> S
    assert restricted_parser.sentence2tree(sentence1).biclosed_type == N

    sentence2 = 'I do'
    assert newccg_parser.sentence2tree(sentence2).biclosed_type == S
    assert restricted_parser.sentence2tree(sentence2).biclosed_type == N
