import pytest

from discopy import biclosed, rigid, Word
from discopy.rigid import Cup, Id, Swap

from discoket.ccg2diagram import CCGTree


@pytest.fixture
def tree():
    n, p, s = biclosed.Ty('n'), biclosed.Ty('p'), biclosed.Ty('s')
    some = CCGTree(text='some', ccg_rule='UNK', biclosed_type=s << p)
    thing = CCGTree(text='thing', ccg_rule='UNK', biclosed_type=n >> p)
    return CCGTree(text='some thing', ccg_rule='FX', biclosed_type=n >> s,
                   children=(some, thing))


def test_biclosed_diagram(tree):
    i = biclosed.Ty()
    n, p, s = biclosed.Ty('n'), biclosed.Ty('p'), biclosed.Ty('s')
    expected_words = (biclosed.Box('some', i, s << p) @
                      biclosed.Box('thing', i, n >> p))
    expected_diagram = expected_words >> biclosed.FX(s << p, n >> p)

    assert tree.to_biclosed_diagram() == expected_diagram


def test_diagram(tree):
    n, p, s = rigid.Ty('n'), rigid.Ty('p'), rigid.Ty('s')
    expected_words = Word('some', s @ p.l) @ Word('thing', n.r @ p)
    expected_diagram = (expected_words >>
                        Id(s) @ Swap(p.l, n.r) @ Id(p) >>
                        Swap(s, n.r) @ Cup(p.l, p))

    assert tree.to_diagram() == expected_diagram
