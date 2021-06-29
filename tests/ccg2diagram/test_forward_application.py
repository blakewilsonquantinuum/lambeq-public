import pytest

from discopy import biclosed, rigid, Word

from discoket.ccg2diagram import CCGTree


@pytest.fixture
def tree():
    n, s = biclosed.Ty('n'), biclosed.Ty('s')
    do = CCGTree(text='do', ccg_rule='UNK', biclosed_type=s << n)
    thing = CCGTree(text='thing', ccg_rule='UNK', biclosed_type=n)
    return CCGTree(text='do thing', ccg_rule='FA', biclosed_type=s,
                   children=(do, thing))


def test_biclosed_diagram(tree):
    i, n, s = biclosed.Ty(), biclosed.Ty('n'), biclosed.Ty('s')
    expected_words = biclosed.Box('do', i, s << n) @ biclosed.Box('thing', i, n)
    expected_diagram = expected_words >> biclosed.FA(s << n)

    assert tree.to_biclosed_diagram() == expected_diagram


def test_diagram(tree):
    n, s = rigid.Ty('n'), rigid.Ty('s')
    expected_words = Word('do', s @ n.l) @ Word('thing', n)
    expected_diagram = expected_words >> (rigid.Id(s) @ rigid.Cup(n.l, n))

    assert tree.to_diagram() == expected_diagram
