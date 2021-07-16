__all__ = ['CCGAtomicType']

from enum import Enum
from typing import Any, List

from discopy import rigid
from discopy.biclosed import Ty

from discoket.core.types import AtomicType


class _CCGAtomicTypeMeta(Ty, Enum):
    def __new__(cls, value: rigid.Ty) -> Ty:
        return object.__new__(Ty)

    def _generate_next_value_(  # type: ignore[override]
            name: str, start: int, count: int, last_values: List[Any]) -> str:
        return AtomicType[name]._value_

    @classmethod
    def conjoinable(cls, _type: Any) -> bool:
        return _type in (cls.CONJUNCTION, cls.PUNCTUATION)


CCGAtomicType = _CCGAtomicTypeMeta('CCGAtomicType',  # type: ignore[call-arg]
                                   [*AtomicType.__members__])
CCGAtomicType.__doc__ = (
        """Standard CCG atomic types mapping to their biclosed type.""")
