import pytest

import json

from discopy.biclosed import Ty

from discoket.ccg2discocat import CCGTree


@pytest.fixture
def tree():
    n, s = Ty('n'), Ty('s')
    do = CCGTree(text='do', ccg_rule='UNK', biclosed_type=(n >> s) << n)
    thing = CCGTree(text='thing', ccg_rule='UNK', biclosed_type=n)
    thing_lexed = CCGTree(text='thing', ccg_rule='LEX', biclosed_type=n,
                          children=(thing,))
    return CCGTree(text='do thing', ccg_rule='FA', biclosed_type=n >> s,
                   children=(do, thing_lexed))


def test_json(tree):
    assert CCGTree.from_json(None) is None
    assert CCGTree.from_json(tree.to_json()) == tree
    assert CCGTree.from_json(json.dumps(tree.to_json())) == tree


def test_properties(tree):
    assert not tree.is_terminal
    assert not tree.is_unary

    assert tree.children[0].is_terminal
    assert not tree.children[0].is_unary

    assert not tree.children[1].is_terminal
    assert tree.children[1].is_unary
