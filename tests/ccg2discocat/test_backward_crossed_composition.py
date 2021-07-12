import pytest

from discopy import biclosed, rigid, Word
from discopy.rigid import Id, Swap

from discoket.ccg2discocat.ccg_tree import PlanarBX, CCGTree


@pytest.fixture
def tree():
    n, p, s = biclosed.Ty('n'), biclosed.Ty('p'), biclosed.Ty('s')
    go = CCGTree(text='go', biclosed_type=(n >> s) << p)
    up = CCGTree(text='up', biclosed_type=(n >> s) >> s)
    return CCGTree(text='go up', ccg_rule='BX', biclosed_type=s << p,
                   children=(go, up))


def test_biclosed_diagram(tree):
    i = biclosed.Ty()
    n, p, s = biclosed.Ty('n'), biclosed.Ty('p'), biclosed.Ty('s')
    expected_words = (biclosed.Box('go', i, (n >> s) << p) @
                      biclosed.Box('up', i, (n >> s) >> s))
    expected_diagram = (expected_words >>
                        biclosed.BX((n >> s) << p, (n >> s) >> s))

    assert tree.to_biclosed_diagram() == expected_diagram


def test_diagram(tree):
    n, p, s = rigid.Ty('n'), rigid.Ty('p'), rigid.Ty('s')
    expected_words = Word('go', n.r @ s @ p.l) @ Word('up', s.r @ n.r.r @ s)
    expected_diagram = (expected_words >>
                        Id(n.r @ s) @ Swap(p.l, s.r) @ Id(n.r.r @ s) >>
                        Id(n.r @ s @ s.r) @ Swap(p.l, n.r.r) @ Id(s) >>
                        rigid.cups(n.r @ s, s.r @ n.r.r) @ Swap(p.l, s))

    assert tree.to_diagram() == expected_diagram


def test_planar_biclosed_diagram(tree):
    i = biclosed.Ty()
    n, p, s = biclosed.Ty('n'), biclosed.Ty('p'), biclosed.Ty('s')
    expected_words = biclosed.Box('go', i, (n >> s) << p)
    up = biclosed.Box('up', i, (n >> s) >> s)
    expected_diagram = (expected_words >>
                        PlanarBX((n >> s) << p, up))

    assert tree.to_biclosed_diagram(planar=True) == expected_diagram


def test_planar_diagram(tree):
    n, p, s = rigid.Ty('n'), rigid.Ty('p'), rigid.Ty('s')
    expected_words = Word('go', n.r @ s @ p.l)
    up = Word('up', s.r @ n.r.r @ s)
    expected_diagram = (expected_words >>
                        Id(n.r @ s) @ up @ Id(p.l) >>
                        rigid.cups(n.r @ s, s.r @ n.r.r) @ Id(s @ p.l))

    assert tree.to_diagram(planar=True) == expected_diagram
