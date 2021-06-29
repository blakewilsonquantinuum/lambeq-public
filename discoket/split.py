"""
Split
=====
A splitter is used to reduce the order of large boxes in DisCoPy
diagrams. This is helpful for reducing the memory requirements for
tensor networks.

"""

__all__ = ['MPSSplitter', 'SpiderSplitter']

from discopy import Word
from discopy.rigid import Box, Cup, Diagram, Functor, Id, Ty

from discoket.core.types import Spider


class MPSSplitter:
    """Split large boxes into matrix product states."""

    BOND_TYPE: Ty = Ty('B')

    def __init__(self, max_order: int = 3, bond_type: Ty = BOND_TYPE) -> None:
        if max_order < 3:
            raise ValueError('`max_order` must be at least 3')
        self.max_order = max_order
        self.bond_type = bond_type
        self.split = Functor(ob=lambda ob: ob, ar=self.ar)

    def __call__(self, diagram: Diagram) -> Diagram:
        return self.split(diagram)

    def ar(self, ar: Word) -> Diagram:
        if len(ar.cod) <= self.max_order:
            return Word(f'{ar.name}_1', ar.cod)

        boxes = []
        cups = []
        step_size = self.max_order - 2
        for i, start in enumerate(range(0, len(ar.cod), step_size)):
            cod = (self.bond_type.r @ ar.cod[start:start+step_size] @
                   self.bond_type)
            boxes.append(Word(f'{ar.name}_{i}', cod))
            cups += [Id(cod[1:-1]), Cup(self.bond_type, self.bond_type.r)]
        boxes[0] = Word(boxes[0].name, boxes[0].cod[1:])
        boxes[-1] = Word(boxes[-1].name, boxes[-1].cod[:-1])

        return Box.tensor(*boxes) >> Diagram.tensor(*cups[:-1])


class SpiderSplitter:
    """Split large boxes into spiders."""

    def __init__(self, max_order: int = 2) -> None:
        if max_order < 2:
            raise ValueError('`max_order` must be at least 2')
        self.max_order = max_order
        self.split = Functor(ob=lambda ob: ob, ar=self.ar)

    def __call__(self, diagram: Diagram) -> Diagram:
        return self.split(diagram)

    def ar(self, ar: Word) -> Diagram:
        if len(ar.cod) <= self.max_order:
            return Word(f'{ar.name}_1', ar.cod)

        boxes = []
        spiders = [Id(ar.cod[:1])]
        step_size = self.max_order - 1
        for i, start in enumerate(range(0, len(ar.cod)-1, step_size)):
            cod = ar.cod[start:start + step_size + 1]
            boxes.append(Word(f'{ar.name}_{i}', cod))
            spiders += [Id(cod[1:-1]), Spider(2, 1, cod[-1:])]
        spiders[-1] = Id(spiders[-1].cod)

        return Diagram.tensor(*boxes) >> Diagram.tensor(*spiders)
