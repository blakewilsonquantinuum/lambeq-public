import pytest

from lambeq.pregroups import create_pregroup_diagram
from lambeq.core.types import AtomicType
from discopy import Cup, Ob, Swap, Ty, Word

n = AtomicType.NOUN
s = AtomicType.SENTENCE


def test_diagram_with_only_cups():
    words = [Word("John", n),
             Word("walks", n.r @ s),
             Word("in", s.r @ n.r.r @ n.r @ s @ n.l),
             Word("the", n @ n.l),
             Word("park", n)]
    cups = [(Cup, 2, 3), (Cup, 7, 8), (Cup, 9, 10), (Cup, 1, 4), (Cup, 0, 5)]
    d = create_pregroup_diagram(words, s, cups)

    expected_boxes = [Word('John', Ty('n')),
                      Word('walks', Ty(Ob('n', z=1), 's')),
                      Word('in', Ty(Ob('s', z=1), Ob('n', z=2), Ob('n', z=1), 's', Ob('n', z=-1))),
                      Word('the', Ty('n', Ob('n', z=-1))),
                      Word('park', Ty('n')), Cup(Ty('s'), Ty(Ob('s', z=1))),
                      Cup(Ty(Ob('n', z=-1)), Ty('n')), Cup(Ty(Ob('n', z=-1)), Ty('n')),
                      Cup(Ty(Ob('n', z=1)), Ty(Ob('n', z=2))),
                      Cup(Ty('n'), Ty(Ob('n', z=1)))]
    expected_offsets = [0, 1, 3, 8, 10, 2, 5, 5, 1, 0]

    assert d.boxes == expected_boxes and d.offsets == expected_offsets


def test_diagram_with_cups_and_swaps():
    words = [Word("John", n),
             Word("gave", n.r @ s @ n.l @ n.l),
             Word("Mary", n),
             Word("a", n @ n.l),
             Word("flower", n)]
    cups = [(Cup, 0, 1), (Swap, 3, 4), (Cup, 4, 5), (Cup, 7, 8), (Cup, 3, 6)]

    d = create_pregroup_diagram(words, s, cups)

    expected_boxes = [Word('John', Ty('n')),
                      Word('gave', Ty(Ob('n', z=1), 's', Ob('n', z=-1), Ob('n', z=-1))),
                      Word('Mary', Ty('n')),
                      Word('a', Ty('n', Ob('n', z=-1))),
                      Word('flower', Ty('n')), Cup(Ty('n'), Ty(Ob('n', z=1))),
                      Swap(Ty(Ob('n', z=-1)), Ty(Ob('n', z=-1))),
                      Cup(Ty(Ob('n', z=-1)), Ty('n')),
                      Cup(Ty(Ob('n', z=-1)), Ty('n')),
                      Cup(Ty(Ob('n', z=-1)), Ty('n'))]
    expected_offsets = [0, 1, 5, 6, 8, 0, 1, 2, 3, 1]
    assert d.boxes == expected_boxes and d.offsets == expected_offsets


def test_diagram_with_a_single_box():
    words = [Word("Yes", s)]
    d = create_pregroup_diagram(words, s)
    assert d.boxes == words and d.offsets == [0]
