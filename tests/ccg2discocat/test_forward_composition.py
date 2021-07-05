import pytest

from discopy import biclosed, rigid, Word

from discoket.ccg2discocat import CCGTree


@pytest.fixture
def tree():
    n, s = biclosed.Ty('n'), biclosed.Ty('s')
    i = CCGTree(text='I', ccg_rule='UNK', biclosed_type=s << s)
    do = CCGTree(text='do', ccg_rule='UNK', biclosed_type=s << n)
    return CCGTree(text='I do', ccg_rule='FC', biclosed_type=s << n,
                   children=(i, do))


def test_biclosed_diagram(tree):
    i, n, s = biclosed.Ty(), biclosed.Ty('n'), biclosed.Ty('s')
    expected_words = (biclosed.Box('I', i, s << s) @
                      biclosed.Box('do', i, s << n))
    expected_diagram = expected_words >> biclosed.FC(s << s, s << n)

    assert tree.to_biclosed_diagram() == expected_diagram


def test_diagram(tree):
    n, s = rigid.Ty('n'), rigid.Ty('s')
    expected_words = Word('I', s @ s.l) @ Word('do', s @ n.l)
    expected_diagram = (expected_words >>
                        rigid.Id(s) @ rigid.Cup(s.l, s) @ rigid.Id(n.l))

    assert tree.to_diagram() == expected_diagram
