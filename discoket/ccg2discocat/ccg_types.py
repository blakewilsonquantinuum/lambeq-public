__all__ = ['CCGAtomicType', 'replace_cat_result']

from enum import Enum
from typing import Any, List, Optional, Tuple

from discopy import rigid
from discopy.biclosed import Ty

from discoket.core.types import AtomicType


class _CCGAtomicTypeMeta(Ty, Enum):
    def __new__(cls, value: rigid.Ty) -> Ty:
        return object.__new__(Ty)

    @staticmethod
    def _generate_next_value_(
            name: str, start: int, count: int, last_values: List[Any]) -> str:
        return AtomicType[name]._value_

    @classmethod
    def conjoinable(cls, _type: Any) -> bool:
        return _type in (cls.CONJUNCTION, cls.PUNCTUATION)


CCGAtomicType = _CCGAtomicTypeMeta('CCGAtomicType',  # type: ignore[call-arg]
                                   [*AtomicType.__members__])
CCGAtomicType.__doc__ = (
        """Standard CCG atomic types mapping to their biclosed type.""")


def replace_cat_result(cat: Ty,
                       original: Ty,
                       replacement: Ty,
                       direction: str = '|') -> Tuple[Ty, Optional[Ty]]:
    """Replace the innermost category result with a new category.

    This attempts to replace provided result category with a replacement. If
    the provided category cannot be found, it replaces the innermost category
    possible. In both cases, the replaced category is returned alongside the
    new category.

    Parameters
    ----------
    cat : discopy.biclosed.Ty
        The category whose result is replaced.
    original : discopy.biclosed.Ty
        The category that should be replaced.
    replacement : discopy.biclosed.Ty
        The replacement for the new category.
    direction : str
        Used to check the operations in the type. Consists of either 1 or 2
        characters, each being one of '/', '\', '|'. If 2 characters, the first
        checks the innermost operation, and the second checks the rest. If only
        1 character, it is used for all checks.

    Returns
    -------
    discopy.biclosed.Ty
        The new category.
    discopy.biclosed.Ty
        The replaced result category.

    """

    if not (len(direction) in (1, 2) and set(direction).issubset(r'\|/')):
        raise ValueError(f'Invalid direction: "{direction}"')
    if not cat.left:
        return cat, None

    cat_dir = '/' if cat == cat.left << cat.right else '\\'
    arg, res = ((cat.right, cat.left) if cat_dir == '/' else
                (cat.left, cat.right))

    basic = res == original or res.left is None
    if not basic:
        if cat_dir != direction[-1] != '|':
            basic = True  # replacing inner categories failed, try this level
        else:
            new, old = replace_cat_result(
                    res, original, replacement, direction)
            if old is None:
                basic = True

    if basic:
        if cat_dir != direction[0] != '|':
            return cat, None
        new, old = replacement, res

    return new << arg if cat_dir == '/' else arg >> new, old
