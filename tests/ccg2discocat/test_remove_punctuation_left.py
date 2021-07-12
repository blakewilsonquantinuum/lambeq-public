import pytest

from discopy import biclosed, rigid, Word

from discoket.ccg2discocat import CCGTree
from discoket.ccg2discocat.ccg_rule import CCGAtomicType, RPL


@pytest.fixture
def tree():
    punc = CCGAtomicType.PUNCTUATION
    s = biclosed.Ty('s')
    comma = CCGTree(text=',', biclosed_type=punc)
    go = CCGTree(text='go', biclosed_type=s)
    return CCGTree(text=', go', ccg_rule='LP', biclosed_type=s,
                   children=(comma, go))


def test_biclosed_diagram(tree):
    punc = CCGAtomicType.PUNCTUATION
    i, s = biclosed.Ty(), biclosed.Ty('s')
    expected_words = biclosed.Box(',', i, punc) @ biclosed.Box('go', i, s)
    expected_diagram = expected_words >> RPL(punc, s)

    assert tree.to_biclosed_diagram() == expected_diagram


def test_diagram(tree):
    expected_diagram = Word('go', rigid.Ty('s'))
    assert tree.to_diagram() == expected_diagram
