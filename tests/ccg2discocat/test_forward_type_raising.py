import pytest

from discopy import biclosed, rigid, Word
from discopy.rigid import Cap, Cup, Id

from discoket.ccg2discocat import CCGTree


@pytest.fixture
def tree():
    n, s = biclosed.Ty('n'), biclosed.Ty('s')
    thing = CCGTree(text='thing', biclosed_type=n)
    return CCGTree(text='thing', ccg_rule='FTR', biclosed_type=s << (n >> s),
                   children=(thing,))


def test_biclosed_diagram(tree):
    i, n, s = biclosed.Ty(), biclosed.Ty('n'), biclosed.Ty('s')
    expected_words = biclosed.Box('thing', i, n)
    expected_diagram = expected_words >> biclosed.Curry(biclosed.BA(n >> s))

    assert tree.to_biclosed_diagram() == expected_diagram


def test_diagram(tree):
    n, s = rigid.Ty('n'), rigid.Ty('s')
    expected_words = Word('thing', n)
    expected_diagram = (expected_words >>
                        Id(n) @ Cap(n.r, n) >>
                        Id(n @ n.r) @ Cap(s, s.l) @ Id(n) >>
                        Cup(n, n.r) @ Id(s @ s.l @ n))

    assert tree.to_diagram() == expected_diagram
