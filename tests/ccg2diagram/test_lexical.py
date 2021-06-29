import pytest

from discopy import biclosed, rigid, Word

from discoket.ccg2diagram import CCGTree


@pytest.fixture
def tree():
    n = biclosed.Ty('n')
    word = CCGTree(text='word', ccg_rule='UNK', biclosed_type=n)
    return CCGTree(text='word', ccg_rule='LEX', biclosed_type=n, children=(word,))


def test_biclosed_diagram(tree):
    expected_diagram = biclosed.Box('word', biclosed.Ty(), biclosed.Ty('n'))
    assert tree.to_biclosed_diagram() == expected_diagram


def test_diagram(tree):
    expected_diagram = Word('word', rigid.Ty('n'))
    assert tree.to_diagram() == expected_diagram
