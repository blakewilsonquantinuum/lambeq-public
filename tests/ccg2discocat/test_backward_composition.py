import pytest

from discopy import biclosed, rigid, Word

from discoket.ccg2discocat import CCGTree


@pytest.fixture
def tree():
    n, s = biclosed.Ty('n'), biclosed.Ty('s')
    do = CCGTree(text='do', biclosed_type=n >> s)
    thing = CCGTree(text='thing', biclosed_type=s >> s)
    return CCGTree(text='do thing', ccg_rule='BC', biclosed_type=n >> s,
                   children=(do, thing))


def test_biclosed_diagram(tree):
    i, n, s = biclosed.Ty(), biclosed.Ty('n'), biclosed.Ty('s')
    expected_words = (biclosed.Box('do', i, n >> s) @
                      biclosed.Box('thing', i, s >> s))
    expected_diagram = expected_words >> biclosed.BC(n >> s, s >> s)

    assert tree.to_biclosed_diagram() == expected_diagram


def test_diagram(tree):
    n, s = rigid.Ty('n'), rigid.Ty('s')
    expected_words = Word('do', n.r @ s) @ Word('thing', s.r @ s)
    expected_diagram = (expected_words >>
                        rigid.Id(n.r) @ rigid.Cup(s, s.r) @ rigid.Id(s))

    assert tree.to_diagram() == expected_diagram
