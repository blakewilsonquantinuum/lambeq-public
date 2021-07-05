import pytest

from discopy import biclosed, rigid, Word

from discoket.ccg2discocat import CCGTree
from discoket.ccg2discocat.ccg_rule import CCGRuleUseError


@pytest.fixture
def tree():
    n = biclosed.Ty('n')
    word = CCGTree(text='word', ccg_rule='UNK', biclosed_type=n)
    return CCGTree(text='word', ccg_rule='UNK', biclosed_type=biclosed.Ty(),
                   children=(word,))


def test_biclosed_diagram(tree):
    with pytest.raises(CCGRuleUseError):
        tree.to_biclosed_diagram()

    expected_diagram = biclosed.Box('word', biclosed.Ty(), biclosed.Ty('n'))
    assert tree.children[0].to_biclosed_diagram() == expected_diagram


def test_diagram(tree):
    with pytest.raises(CCGRuleUseError):
        tree.to_diagram()

    expected_diagram = Word('word', rigid.Ty('n'))
    assert tree.children[0].to_diagram() == expected_diagram
