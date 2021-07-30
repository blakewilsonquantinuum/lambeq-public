__all__ = ['CCGAtomicType', 'CCGParseError', 'replace_cat_result',
           'str2biclosed']

from enum import Enum
from typing import Any, Callable, List, Optional, Tuple

from discopy import rigid
from discopy.biclosed import Ty

from discoket.core.types import AtomicType

CONJ_TAG = '[conj]'


class CCGParseError(Exception):
    """Error when parsing a CCG type string."""


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


def str2biclosed(cat: str, str2type: Callable[[str], Ty] = Ty) -> Ty:
    r"""Parse a CCG category string into a biclosed type.

    The string should follow the following grammar:
        atomic_cat    = { <any character except "(", ")", "/" and "\"> }
        op            = "/" | "\"
        bracketed_cat = atomic_cat | "(" bracketed_cat [ op bracketed_cat ] ")"
        cat           = bracketed_cat [ op bracketed_cat ] [ "[conj]" ]

    Parameters
    ----------
    cat : str
        The string to be parsed.
    str2type: callable
        A function that parses an atomic category into a biclosed type. The
        default uses `discopy.biclosed.Ty` to produce a type with the same
        name as the atomic category.

    Returns
    -------
    discopy.biclosed.Ty
        The parsed category as a biclosed type.

    Raises
    ------
    CCGParseError
        If parsing fails.

    Notes
    -----
    Conjunctions follow the CCGBank convention of:
      x   and  y
      C  conj  C
       \    \ /
        \ C[conj]
         \ /
          C
    thus equivalent to C\C.

    """

    if cat.endswith(CONJ_TAG):
        clean_cat = f'({cat[:-len(CONJ_TAG)]})'
        base_type, end = _clean_str2biclosed(clean_cat, str2type)
        biclosed_type = base_type >> base_type
    else:
        clean_cat = f'({cat})'
        biclosed_type, end = _clean_str2biclosed(clean_cat, str2type)
    if end != len(clean_cat):
        raise CCGParseError(f'Failed to parse "{cat}": extra text after '
                            f'character {end} - "{cat[end-1:]}".')
    return biclosed_type


def _clean_str2biclosed(cat: str,
                        str2type: Callable[[str], Ty] = Ty,
                        start: int = 0) -> Tuple[Ty, int]:
    if cat[start] != '(':
        # base case
        end = start + 1
        while end < len(cat):
            if cat[end] in r'/\)':
                break
            end += 1
        biclosed_type = str2type(cat[start:end])
    else:
        biclosed_type, end = _clean_str2biclosed(cat, str2type, start + 1)
        op = cat[end]
        if op in r'\/':
            right, end = _clean_str2biclosed(cat, str2type, end + 1)
            biclosed_type = (biclosed_type << right if op == '/' else
                             right >> biclosed_type)
        if cat[end] != ')':
            raise CCGParseError(
                    f'Failed to parse "{cat}": unmatched "(" at character '
                    f'{start+1}, expected at character {end+1}.')
        end += 1
    return biclosed_type, end


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
        Used to check the operations in the category. Consists of either 1 or 2
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

    # `replace` indicates whether `res` should be replaced, due to one of the
    # following conditions being true:
    # - `res` matches `original`
    # - `res` is an atomic type
    # - `cat_dir` does not match the required operation
    # - attempting to replace any inner category fails
    replace = res == original or res.left is None
    if not replace:
        if cat_dir != direction[-1] != '|':
            replace = True
        else:
            new, old = replace_cat_result(
                    res, original, replacement, direction)
            if old is None:
                replace = True  # replacing inner category failed

    if replace:
        if cat_dir != direction[0] != '|':
            return cat, None
        new, old = replacement, res

    return new << arg if cat_dir == '/' else arg >> new, old
