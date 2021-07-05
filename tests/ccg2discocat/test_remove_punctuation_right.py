import pytest

from discopy import biclosed, rigid, Word

from discoket.ccg2discocat import CCGTree
from discoket.ccg2discocat.ccg_rule import CCGAtomicType, RPR


@pytest.fixture
def tree():
    punc = CCGAtomicType.PUNCTUATION
    s = biclosed.Ty('s')
    go = CCGTree(text='go', ccg_rule='UNK', biclosed_type=s)
    comma = CCGTree(text=',', ccg_rule='UNK', biclosed_type=punc)
    return CCGTree(text='go ,', ccg_rule='RP', biclosed_type=s,
                   children=(go, comma))


def test_biclosed_diagram(tree):
    punc = CCGAtomicType.PUNCTUATION
    i, s = biclosed.Ty(), biclosed.Ty('s')
    expected_words = biclosed.Box('go', i, s) @ biclosed.Box(',', i, punc)
    expected_diagram = expected_words >> RPR(s, punc)

    assert tree.to_biclosed_diagram() == expected_diagram


def test_diagram(tree):
    expected_diagram = Word('go', rigid.Ty('s'))
    assert tree.to_diagram() == expected_diagram
