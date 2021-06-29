__all__ = ['AtomicType', 'Spider']

from enum import Enum

from discopy.rigid import Box, Ty


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


class Spider(Box):
    """Spider for rigid diagrams."""

    def __init__(self, n_legs_in: int, n_legs_out: int, _type: Ty) -> None:
        name = 'Spider({}, {}, {})'.format(n_legs_in, n_legs_out, _type)
        dom, cod = _type ** n_legs_in, _type ** n_legs_out
        super().__init__(name, dom, cod, draw_as_spider=len(_type) == 1,
                         color='black', drawing_name='')
        self.type = _type

    def __repr__(self) -> str:
        return self.name

    def dagger(self) -> 'Spider':
        return type(self)(len(self.cod), len(self.dom), self.type)

    @property
    def dim(self) -> Ty:
        return self.type
