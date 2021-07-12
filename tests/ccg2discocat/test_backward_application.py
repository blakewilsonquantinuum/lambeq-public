import pytest

from discopy import biclosed, rigid, Word

from discoket.ccg2discocat import CCGTree


@pytest.fixture
def tree():
    n, s = biclosed.Ty('n'), biclosed.Ty('s')
    i = CCGTree(text='I', biclosed_type=n)
    do = CCGTree(text='do', biclosed_type=n >> s)
    return CCGTree(text='I do', ccg_rule='BA', biclosed_type=s,
                   children=(i, do))


def test_biclosed_diagram(tree):
    i, n, s = biclosed.Ty(), biclosed.Ty('n'), biclosed.Ty('s')
    expected_words = biclosed.Box('I', i, n) @ biclosed.Box('do', i, n >> s)
    expected_diagram = expected_words >> biclosed.BA(n >> s)

    assert tree.to_biclosed_diagram() == expected_diagram


def test_diagram(tree):
    n, s = rigid.Ty('n'), rigid.Ty('s')
    expected_words = Word('I', n) @ Word('do', n.r @ s)
    expected_diagram = expected_words >> (rigid.Cup(n, n.r) @ rigid.Id(s))

    assert tree.to_diagram() == expected_diagram
