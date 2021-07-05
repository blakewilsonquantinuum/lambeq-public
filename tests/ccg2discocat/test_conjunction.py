import pytest

from discopy import biclosed, rigid, Word

from discoket.ccg2discocat import CCGTree
from discoket.ccg2discocat.ccg_rule import CCGAtomicType


@pytest.fixture
def trees():
    conj = CCGAtomicType.CONJUNCTION
    n = biclosed.Ty('n')
    alice = CCGTree(text='alice', ccg_rule='UNK', biclosed_type=n)
    plus = CCGTree(text='plus', ccg_rule='UNK', biclosed_type=conj)

    tree_left = CCGTree(text='plus alice', ccg_rule='CONJ',
                        biclosed_type=n >> n, children=(plus, alice))
    tree_right = CCGTree(text='alice plus', ccg_rule='CONJ',
                         biclosed_type=n << n, children=(alice, plus))
    return (tree_left, tree_right)


def test_biclosed_diagram_left(trees):
    i, n = biclosed.Ty(), biclosed.Ty('n')
    expected_words = (biclosed.Box('plus', i, (n >> n) << n) @
                      biclosed.Box('alice', i, n))
    expected_diagram = expected_words >> biclosed.FA((n >> n) << n)

    assert trees[0].to_biclosed_diagram() == expected_diagram


def test_diagram_left(trees):
    n = rigid.Ty('n')
    expected_words = Word('plus', n.r @ n @ n.l) @ Word('alice', n)
    expected_diagram = expected_words >> (rigid.Id(n.r @ n) @ rigid.Cup(n.l, n))

    assert trees[0].to_diagram() == expected_diagram


def test_biclosed_diagram_right(trees):
    i, n = biclosed.Ty(), biclosed.Ty('n')
    expected_words = (biclosed.Box('alice', i, n) @
                      biclosed.Box('plus', i, n >> (n << n)))
    expected_diagram = expected_words >> biclosed.BA(n >> (n << n))

    assert trees[1].to_biclosed_diagram() == expected_diagram


def test_diagram_right(trees):
    n = rigid.Ty('n')
    expected_words = Word('alice', n) @ Word('plus', n.r @ n @ n.l)
    expected_diagram = expected_words >> (rigid.Cup(n, n.r) @ rigid.Id(n @ n.l))

    assert trees[1].to_diagram() == expected_diagram
