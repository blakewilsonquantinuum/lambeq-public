__all__ = ['AtomicType']

from enum import Enum

from discopy.rigid import Ty


class AtomicType(Ty, Enum):
    """Standard CCG atomic types mapping to their rigid type."""

    def __new__(_, value: str) -> Ty:
        return Ty(value)

    NOUN = 'n'
    NOUN_PHRASE = 'n'
    SENTENCE = 's'
    PREPOSITION = 'p'
    CONJUNCTION = 'conj'
    PUNCTUATION = 'punc'
