__all__ = ['AtomicType']

from enum import Enum

from discopy.rigid import Ty


class AtomicType(Ty, Enum):
    """Standard pregroup atomic types mapping to their rigid type."""

    def __new__(cls, value: str) -> Ty:
        return object.__new__(Ty)

    NOUN = 'n'
    NOUN_PHRASE = 'n'
    SENTENCE = 's'
    PREPOSITION = 'p'
    CONJUNCTION = 'conj'
    PUNCTUATION = 'punc'
