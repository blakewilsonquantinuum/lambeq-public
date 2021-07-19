from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterator, List, Tuple

import discopy

from discoket.ccg2discocat.ccg_rule import CCGRule, CCGAtomicType


class CCGBankNodeType(str, Enum):
    TREE = 'T'
    LEXICAL = 'L'


@dataclass
class CCGBankNode:
    type: str = ""
    cat: str = ""
    token: str = ""
    head_id: int = 0
    num_children: int = 0


def read_ccgbank_section(path: Path, section_id: int) -> Iterator[Tuple[str, str]]:
    """Read a CCGBank section and return (id, tree) tuples."""
    path = path / str(section_id).zfill(2)
    cur_id = ""
    for auto_file in path.iterdir():
        with open(auto_file, "r") as f:
            lines = f.readlines()
        for ln in lines:
            if ln.startswith("ID="):
                cur_id = ln.split()[0][3:]
            else:
                yield cur_id, ln.strip()


def ccg_to_biclosed(cat: str) -> discopy.biclosed.Ty:
    """Transform a CCGBank category into a biclosed type."""

    def get_parts(cat: str) -> Tuple[str, str, str]:
        par_count = 0
        for idx, ch in enumerate(cat):
            if ch == "(":
                par_count += 1
            elif ch == ")":
                par_count -= 1
            elif ch in ["/", "\\"]:
                if par_count == 0:
                    left = cat[0:idx]
                    right = cat[idx + 1:]

                    if left[0] == '(':
                        left = left[1:-1]
                    if right[0] == '(':
                        right = right[1:-1]
                    return left, right, ch
        return '', '', ''

    # Treatment of conjunctions. CCGBank uses the following scheme:
    #          NP
    #        /     \
    #       /    NP[conj]
    #      /      /    \
    #     /    conj    NP
    #  apples  and   oranges
    if cat.endswith("[conj]"):
        clean_cat = cat[:-6]
        if '/' in clean_cat or '\\' in clean_cat:
            cat = f"({clean_cat})\\({clean_cat})"
        else:
            cat = f"{clean_cat}\\{clean_cat}"

    is_functor = '/' in cat or '\\' in cat
    if not is_functor:
        if cat in ["N", "NP"] or cat.startswith('N[') or cat.startswith('NP['):
            return CCGAtomicType.NOUN
        if cat == "S" or cat.startswith('S['):
            return CCGAtomicType.SENTENCE
        if cat == 'PP' or cat.startswith('PP['):
            return CCGAtomicType.PREPOSITION
        if cat == 'conj':
            return CCGAtomicType.CONJUNCTION
        if cat in ['LRB', 'RRB'] or cat in ',.:;':
            return CCGAtomicType.PUNCTUATION
    else:
        left, right, slash = get_parts(cat)
        if slash == '/':
            return ccg_to_biclosed(left) << ccg_to_biclosed(right)
        if slash == '\\':
            return ccg_to_biclosed(right) >> ccg_to_biclosed(left)
    raise Exception(f'Invalid CCG type: {cat}')


def determine_ccg_rule(parent: discopy.biclosed.Ty, children: List[discopy.biclosed.Ty]) -> CCGRule:
    """Determine the CCG rule based on the parent type and the children types.

    Called only for CCGBank <T> nodes (not <L> nodes).
    """

    rule = CCGRule.UNKNOWN

    if len(children) == 2:

        child1, child2 = children

        # Forward application
        if (
            isinstance(child1, discopy.biclosed.Over) and
            child1.right == child2 and
            parent == child1.left
        ):
            rule = CCGRule.FORWARD_APPLICATION

        # Backward application
        if (
            isinstance(child2, discopy.biclosed.Under) and
            child1 == child2.left and
            parent == child2.right
        ):
            rule = CCGRule.BACKWARD_APPLICATION

        # Forward harmonic composition
        if (
            type(child1) == type(child2) == discopy.biclosed.Over and
            child1.right == child2.left and
            parent == child1.left << child2.right
        ):
            rule = CCGRule.FORWARD_COMPOSITION

        # Backward harmonic composition
        if (
            type(child1) == type(child2) == discopy.biclosed.Under and
            child1.right == child2.left and
            parent == child1.left >> child2.right
        ):
            rule = CCGRule.BACKWARD_COMPOSITION

        # Forward crossed composition
        if (
            isinstance(child1, discopy.biclosed.Over) and
            isinstance(child2, discopy.biclosed.Under) and
            parent == child2.left >> child1.left
        ):
            rule = CCGRule.FORWARD_CROSSED_COMPOSITION

        # Backward crossed composition
        if (
            isinstance(child1, discopy.biclosed.Over) and
            isinstance(child2, discopy.biclosed.Under) and
            parent == child2.right << child1.right
        ):
            rule = CCGRule.BACKWARD_CROSSED_COMPOSITION

        # Conjunction
        if parent == CCGAtomicType.CONJUNCTION or child1 == CCGAtomicType.CONJUNCTION:
            rule = CCGRule.CONJUNCTION

        # Punctuation
        if child1 == CCGAtomicType.PUNCTUATION:
            rule = CCGRule.REMOVE_PUNCTUATION_LEFT
        if child2 == CCGAtomicType.PUNCTUATION:
            rule = CCGRule.REMOVE_PUNCTUATION_RIGHT

    else:  # Unary rule

        child = children[0]

        # Forward type-raising
        if (
            isinstance(parent, discopy.biclosed.Over) and
            isinstance(parent.right, discopy.biclosed.Under) and
            child == parent.right.left and
            parent.left == parent.right.right
        ):
            rule = CCGRule.FORWARD_TYPE_RAISING

        # Backward type-raising
        elif (
            isinstance(parent, discopy.biclosed.Under) and
            isinstance(parent.left, discopy.biclosed.Over) and
            child == parent.left.right and
            parent.right == parent.left.left
        ):
            rule = CCGRule.BACKWARD_TYPE_RAISING

        else:
            rule = CCGRule.UNARY

    return rule
