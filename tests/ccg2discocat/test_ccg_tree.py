import pytest

import json

from discopy.biclosed import Ty

from discoket.ccg2discocat import CCGTree


@pytest.fixture
def tree():
    n, s = Ty('n'), Ty('s')
    do = CCGTree(text='do', biclosed_type=(n >> s) << n)
    thing = CCGTree(text='thing', biclosed_type=n)
    thing_unary = CCGTree(text='thing', rule='U', biclosed_type=n, children=(thing,))
    return CCGTree(text='do thing', rule='FA', biclosed_type=n >> s,
                   children=(do, thing_unary))


def test_child_reqs(tree):
    with pytest.raises(ValueError):
        CCGTree(rule='U', biclosed_type=tree.biclosed_type, children=tree.children)


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
