# Copyright 2021-2023 Cambridge Quantum Computing Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Grammar category
================
Lambeq's internal representation of the grammar category. This work is
based on DisCoPy (https://discopy.org/) which is released under the
BSD 3-Clause "New" or "Revised" License.

"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field, replace
from typing import Any, ClassVar, Type, TypeVar
from typing import cast, overload, TYPE_CHECKING

from typing_extensions import Self


@dataclass
class Entity:
    category: ClassVar[Category]


_EntityType = TypeVar('_EntityType', bound=Type[Entity])


@dataclass
class Category:
    """The base class for all categories."""
    name: str
    Ty: type[Ty] = field(init=False)
    Box: type[Box] = field(init=False)
    Layer: type[Layer] = field(init=False)
    Diagram: type[Diagram] = field(init=False)

    def set(self, name: str, entity: _EntityType) -> _EntityType:
        setattr(self, name, entity)
        entity.category = self
        return entity

    @overload
    def __call__(self, name_or_entity: str) -> Callable[[_EntityType],
                                                        _EntityType]:
        ...

    @overload
    def __call__(self, name_or_entity: _EntityType) -> _EntityType: ...

    def __call__(
        self,
        name_or_entity: _EntityType | str
    ) -> _EntityType | Callable[[_EntityType], _EntityType]:
        if isinstance(name_or_entity, str):
            name = name_or_entity

            def set_(entity: _EntityType) -> _EntityType:
                return self.set(name, entity)
            return set_
        else:
            return self.set(name_or_entity.__name__, name_or_entity)


grammar = Category('grammar')


@grammar
@dataclass
class Ty(Entity):
    """A type in the grammar category.

    Every type is either atomic, complex, or empty. Complex types are
    tensor products of atomic types, and empty types are the identity
    type.

    Parameters
    ----------
    name : str, optional
        The name of the type, by default None
    objects : list[Ty], optional
        The objects defining a complex type, by default []
    z : int, optional
        The winding number of the type, by default 0


    """
    name: str | None = None
    objects: list[Self] = field(default_factory=list)
    z: int = 0

    category: ClassVar[Category]

    def __post_init__(self) -> None:
        assert len(self.objects) != 1
        assert not (len(self.objects) > 1 and self.name is not None)
        if not self.is_atomic:
            assert self.z == 0

    def is_adjoint(self, other: Ty) -> bool:
        return self == other.l or self == other.r

    @property
    def is_empty(self) -> bool:
        return not self.objects and self.name is None

    @property
    def is_atomic(self) -> bool:
        return not self.objects and self.name is not None

    @property
    def is_complex(self) -> bool:
        return bool(self.objects)

    def __repr__(self) -> str:
        if self.is_empty:
            return 'Ty()'
        elif self.is_atomic:
            return f'Ty({self.name}){".l"*(-self.z)}{".r"*self.z}'
        else:
            return ' @ '.join(map(repr, self.objects))

    def __str__(self) -> str:
        if self.is_empty:
            return 'Ty()'
        elif self.is_atomic:
            return f'{self.name}{".l"*(-self.z)}{".r"*self.z}'
        else:
            return ' @ '.join(map(str, self.objects))

    def __len__(self) -> int:
        return 1 if self.is_atomic else len(self.objects)

    def __iter__(self) -> Iterator[Self]:
        if self.is_atomic:
            yield self
        else:
            yield from self.objects

    def __getitem__(self, index: int | slice) -> Self:
        objects = [*self]
        if TYPE_CHECKING:
            objects = cast(list[Self], objects)
        if isinstance(index, int):
            return objects[index]
        else:
            return self._fromiter(objects[index])

    @classmethod
    def _fromiter(cls, objects: Iterable[Self]) -> Self:
        """Create a Ty from an iterable of atomic objects."""
        objects = list(objects)
        if not objects:
            return cls()
        elif len(objects) == 1:
            return objects[0]
        else:
            return cls(objects=objects)  # type: ignore[arg-type]

    def tensor(self, *tys: Self) -> Self:
        if any(not isinstance(ty, type(self)) for ty in tys):
            return NotImplemented

        return self._fromiter(ob for ty in (self, *tys) for ob in ty)

    def __matmul__(self, rhs: Self) -> Self:
        return self.tensor(rhs)

    def wind(self, z: int) -> Self:
        if self.is_empty or z == 0:
            return self
        elif self.is_atomic:
            return replace(self, z=self.z + z)
        else:
            objects = reversed(self.objects) if z % 2 == 1 else self.objects
            return type(self)(objects=[ob.wind(z) for ob in objects])

    def unwind(self) -> Self:
        return self.wind(-self.z)

    @property
    def l(self) -> Self:  # noqa: E741, E743
        return self.wind(-1)

    @property
    def r(self) -> Self:
        return self.wind(1)

    def __lshift__(self, rhs: Self) -> Self:
        if not isinstance(rhs, type(self)):
            return NotImplemented
        return self @ rhs.l

    def __rshift__(self, rhs: Self) -> Self:
        if not isinstance(rhs, type(self)):
            return NotImplemented
        return self.r @ rhs

    def repeat(self, times: int) -> Self:
        assert times >= 0
        return type(self)().tensor(*[self] * times)

    def __pow__(self, times: int) -> Self:
        return self.repeat(times)


@grammar
@dataclass
class Box(Entity):
    """A box in the grammar category.

    Parameters
    ----------
    name : str
        The name of the box.
    dom : Ty
        The domain of the box.
    cod : Ty
        The codomain of the box.
    z : int, optional
        The winding number of the box, by default 0

    """
    name: str
    dom: Ty
    cod: Ty
    z: int = 0

    def __repr__(self) -> str:
        return (f'[{self.name}{".l"*(-self.z)}{".r"*self.z}; '
                f'{repr(self.dom)} -> {repr(self.cod)}]')

    def __str__(self) -> str:
        return f'{self.name}{".l"*(-self.z)}{".r"*self.z}'

    def to_diagram(self) -> Diagram:
        ID = self.category.Ty()
        return self.category.Diagram(dom=self.dom,
                                     cod=self.cod,
                                     layers=[self.category.Layer(box=self,
                                                                 left=ID,
                                                                 right=ID)])

    def __getattr__(self, name: str) -> Any:
        return getattr(self.to_diagram(), name)

    def __matmul__(self, rhs: Box | Diagram) -> Diagram:
        return self.to_diagram().tensor(rhs.to_diagram())

    def __rshift__(self, rhs: Box | Diagram) -> Diagram:
        return self.to_diagram().then(rhs.to_diagram())

    def wind(self, z: int) -> Self:
        return replace(self,
                       dom=self.dom.wind(z),
                       cod=self.cod.wind(z),
                       z=self.z + z)

    @property
    def l(self) -> Self:  # noqa: E741, E743
        return self.wind(-1)

    @property
    def r(self) -> Self:
        return self.wind(1)

    def unwind(self) -> Self:
        return self.wind(-self.z)

    def dagger(self) -> Daggered | Box:
        return Daggered(self)


@grammar
@dataclass
class Layer(Entity):
    """A layer in a diagram.

    Parameters
    ----------
    box : Box
        The box in the layer.
    left : Ty
        The wire type to the left of the box.
    right : Ty
        The wire type to the right of the box.

    """
    box: Box
    left: Ty
    right: Ty

    def __repr__(self) -> str:
        return f'|{repr(self.left)} @ {repr(self.box)} @ {repr(self.right)}|'

    def extend(self,
               left: Ty | None = None,
               right: Ty | None = None) -> Self:
        ID = self.category.Ty()
        if left is None:
            left = ID
        if right is None:
            right = ID
        return replace(self, left=left @ self.left, right=self.right @ right)


@grammar
@dataclass
class Diagram(Entity):
    """A diagram in the grammar category.

    Parameters
    ----------
    dom : Ty
        The type of the input wires.
    cod : Ty
        The type of the output wires.
    layers : list[Layer]
        The layers of the diagram.

    """
    dom: Ty
    cod: Ty
    layers: list[Layer]

    def __repr__(self) -> str:
        if self.is_id:
            return f'Id({repr(self.dom)})'
        else:
            return ' >> '.join(map(repr, self.layers))

    def to_diagram(self) -> Self:
        return self

    @classmethod
    def id(cls, dom: Ty | None = None) -> Self:
        if dom is None:
            dom = cls.category.Ty()
        return cls(dom=dom, cod=dom, layers=[])

    @property
    def is_id(self) -> bool:
        return not self.layers

    @property
    def boxes(self) -> list[Box]:
        return [layer.box for layer in self.layers]

    @classmethod
    def create_pregroup_diagram(
        cls,
        words: list[Word],
        morphisms: list[tuple[type, int, int]]
    ) -> Self:
        """Create a :py:class:`~.Diagram` from cups and swaps.

            >>> n, s = Ty('n'), Ty('s')
            >>> words = [Word('she', n), Word('goes', n.r @ s @ n.l),
            ...          Word('home', n)]
            >>> morphs = [(Cup, 0, 1), (Cup, 3, 4)]
            >>> diagram = Diagram.create_pregroup_diagram(words, morphs)

        Parameters
        ----------
        words : list of :py:class:`~lambeq.backend.Word`
            A list of :py:class:`~lambeq.backend.Word` s
            corresponding to the words of the sentence.
        morphisms: list of tuple[type, int, int]
            A list of tuples of the form:
                (morphism, start_wire_idx, end_wire_idx).
            Morphisms can be :py:class:`~lambeq.backend.Cup` s or
            :py:class:`~lambeq.backend.Swap` s, while the two numbers
            define the indices of the wires on which the morphism is
            applied.

        Returns
        -------
        :py:class:`~lambeq.backend..Diagram`
            The generated pregroup diagram.

        Raises
        ------
        ValueError
            If the provided morphism list does not type-check properly.

        """
        types: Ty = cls.category.Ty()
        boxes: list[Word] = []
        offsets: list[int] = []
        for w in words:
            boxes.append(w)
            offsets.append(len(types))
            types @= w.cod

        for idx, (typ, start, end) in enumerate(morphisms):
            if typ not in (Cup, Swap):
                raise ValueError(f'Unknown morphism type: {typ}')
            box = typ(types[start:start+1], types[end:end+1])

            boxes.append(box)
            actual_idx = start
            for pr_idx in range(idx):
                if (morphisms[pr_idx][0] == Cup
                        and morphisms[pr_idx][1] < start):
                    actual_idx -= 2
            offsets.append(actual_idx)

        boxes_and_offsets = list(zip(boxes, offsets))
        diagram = cls.id()
        for box, offset in boxes_and_offsets:
            left = diagram.cod[:offset]
            right = diagram.cod[offset + len(box.dom):]
            diagram = diagram >> cls.id(left) @ box @ cls.id(right)
        return diagram

    def cup(self, *pos: int):
        """Apply a cup to the diagram."""
        raise NotImplementedError  # TODO in new PR

    def is_pregroup(self) -> bool:
        """Check if a diagram is a pregroup diagram.

        Adapted from :py:class:`discopy.grammar.pregroup.draw`.

        Returns
        -------
        bool
            Whether the diagram is a pregroup diagram.

        """

        in_words = True
        for layer in self.layers:
            if in_words and isinstance(layer.box, Word):
                if not layer.right.is_empty:
                    return False
            else:
                if not isinstance(layer.box, (Cup, Swap)):
                    return False
                in_words = False
        return True

    def tensor(self, *diagrams: Self) -> Self:
        try:
            diags = [diagram.to_diagram() for diagram in diagrams]
        except AttributeError:
            return NotImplemented
        if any(not isinstance(diagram, type(self)) for diagram in diags):
            return NotImplemented

        right = dom = self.dom.tensor(*[diagram.dom for diagram in diagrams])
        left = self.category.Ty()
        layers = []
        for diagram in (self, *diags):
            right = right[len(diagram.dom):]
            layers += [layer.extend(left, right) for layer in diagram.layers]
            left @= diagram.cod

        return type(self)(dom=dom, cod=left, layers=layers)

    @property
    def offsets(self) -> list[int]:
        """ The offset of a box is the length of the type on its left. """
        return [len(layer.left) for layer in self.layers]

    def __matmul__(self, rhs: Self) -> Self:
        return self.tensor(rhs)

    def __iter__(self) -> Iterator[Layer]:
        yield from self.layers

    def __len__(self) -> int:
        return len(self.layers)

    def then(self, *diagrams: Self) -> Self:
        try:
            diags = [diagram.to_diagram() for diagram in diagrams]
        except AttributeError:
            return NotImplemented
        if any(not isinstance(diagram, type(self)) for diagram in diags):
            return NotImplemented

        layers = [*self.layers]
        cod = self.cod
        for n, diagram in enumerate(diags):
            if diagram.dom != cod:
                raise ValueError(f'Diagram {n} (cod={cod}) does not compose '
                                 f'with diagram {n+1} (dom={diagram.dom})')
            cod = diagram.cod

            layers.extend(diagram.layers)

        return type(self)(dom=self.dom, cod=cod, layers=layers)

    def __rshift__(self, rhs: Self) -> Self:
        return self.then(rhs)

    @property
    def l(self) -> Self:  # noqa: E741, E743
        return type(self)(dom=self.dom.l,
                          cod=self.cod.l,
                          layers=[type(layer)(box=layer.box.l,
                                              left=layer.right.l,
                                              right=layer.left.l)
                                  for layer in self.layers])

    @property
    def r(self) -> Self:
        return type(self)(dom=self.dom.r,
                          cod=self.cod.r,
                          layers=[type(layer)(box=layer.box.r,
                                              left=layer.right.r,
                                              right=layer.left.r)
                                  for layer in self.layers])

    def dagger(self) -> Self:
        if self.is_id:
            return self
        else:
            return type(self)(dom=self.cod,
                              cod=self.dom,
                              layers=[replace(layer, box=layer.box.dagger())
                                      for layer in reversed(self.layers)])

    def normal_form(self):
        raise NotImplementedError  # TODO in new PR

    def draw(self, **kwargs):
        raise NotImplementedError  # TODO in new PR


@dataclass
class Cap(Box):
    """The unit of the adjunction for an atomic type.

    Parameters
    ----------
    left : Ty
        The atomic type.
    right : Ty
        Its left adjoint.

    """
    left: Ty
    right: Ty

    name: str = field(init=False)
    dom: Ty = field(init=False)
    cod: Ty = field(init=False)
    z: int = field(init=False)

    def __post_init__(self) -> None:
        if not self.left.is_atomic or not self.right.is_atomic:
            raise ValueError('left and right need to be atomic types.')
        if not self.left.is_adjoint(self.right):
            raise ValueError('left and right need to be adjoints')
        self.name = 'CAP'
        self.dom = self.category.Ty()
        self.cod = self.left @ self.right
        self.z = 0

    @property
    def l(self) -> Self:  # noqa: E741, E743
        return type(self)(self.right.l, self.left.l)

    @property
    def r(self) -> Self:
        return type(self)(self.right.r, self.left.r)

    def dagger(self) -> Cup:
        return Cup(self.left, self.right)

    __repr__ = Box.__repr__


@dataclass
class Cup(Box):
    """The counit of the adjunction for an atomic type.

    Parameters
    ----------
    left : Ty
        The atomic type.
    right : Ty
        Its left adjoint.

    """
    left: Ty
    right: Ty

    name: str = field(init=False)
    dom: Ty = field(init=False)
    cod: Ty = field(init=False)
    z: int = field(init=False)

    def __post_init__(self) -> None:
        if not self.left.is_atomic or not self.right.is_atomic:
            raise ValueError('left and right need to be atomic types.')
        if not self.left.is_adjoint(self.right):
            raise ValueError('left and right need to be adjoints')
        self.name = 'CUP'
        self.dom = self.left @ self.right
        self.cod = self.category.Ty()
        self.z = 0

    @property
    def l(self) -> Self:  # noqa: E741, E743
        return type(self)(self.right.l, self.left.l)

    @property
    def r(self) -> Self:
        return type(self)(self.right.r, self.left.r)

    def dagger(self) -> Cap:
        return Cap(self.left, self.right)

    __repr__ = Box.__repr__


@dataclass
class Daggered(Box):
    """A daggered box.

    Parameters
    ----------
    box : Box
        The box to be daggered.

    """
    box: Box
    name: str = field(init=False)
    dom: Ty = field(init=False)
    cod: Ty = field(init=False)
    z: int = field(init=False)

    def __post_init__(self) -> None:
        self.name = self.box.name + '†'
        self.dom = self.box.cod
        self.cod = self.box.dom
        self.z = 0

    def dagger(self) -> Box | Daggered:
        return self.box

    @property
    def l(self) -> Self:  # noqa: E741, E743
        return type(self)(self.box.l)

    @property
    def r(self) -> Self:
        return type(self)(self.box.r)

    __repr__ = Box.__repr__


@dataclass
class Spider(Box):
    """A spider in the grammar category.

    Parameters
    ----------
    type : Ty
        The atomic type of the spider.
    n_legs_in : int
        The number of input legs.
    n_legs_out : int
        The number of output legs.

    """
    type: Ty
    n_legs_in: int
    n_legs_out: int

    name: str = field(init=False)
    dom: Ty = field(init=False)
    cod: Ty = field(init=False)
    z: int = field(init=False)

    def __post_init__(self) -> None:
        if not self.type.is_atomic:
            raise TypeError('Spider type needs to be atomic.')
        self.name = 'SPIDER'
        self.dom = self.type ** self.n_legs_in
        self.cod = self.type ** self.n_legs_out
        self.z = 0

    @property
    def l(self) -> Self:  # noqa: E741, E743
        return type(self)(self.type.l, len(self.dom), len(self.cod))

    @property
    def r(self) -> Self:
        return type(self)(self.type.r, len(self.dom), len(self.cod))

    def dagger(self) -> Self:
        return type(self)(self.type, self.n_legs_out, self.n_legs_in)

    __repr__ = Box.__repr__


@dataclass
class Swap(Box):
    """A swap in the grammar category.

    Swaps two wires.

    Parameters
    ----------
    left : Ty
        The atomic type of the left input wire.
    right : Ty
        The atomic type of the right input wire.

    """
    left: Ty
    right: Ty

    name: str = field(init=False)
    dom: Ty = field(init=False)
    cod: Ty = field(init=False)
    z: int = field(init=False)

    def __post_init__(self) -> None:
        if not self.left.is_atomic or not self.right.is_atomic:
            raise ValueError('Types need to be atomic')

        self.name = f'Swap({repr(self.left)}, {repr(self.right)})'
        self.dom, self.cod = self.left @ self.right, self.right @ self.left

    def dagger(self) -> Self:
        return type(self)(self.right, self.left)

    @property
    def l(self) -> Self:  # noqa: E741, E743
        return type(self)(self.right.l, self.left.l)

    @property
    def r(self) -> Self:
        return type(self)(self.right.r, self.left.r)

    __repr__ = Box.__repr__


@dataclass
class Word(Box):
    """A word in the grammar category.

    A word is a :py:class:`~.Box` with an empty domain.

    Parameters
    ----------
    name : str
        The name of the word.
    cod : Ty
        The codomain of the word.
    z : int, optional
        The winding number of the word, by default 0

    """
    name: str
    cod: Ty

    dom: Ty = field(init=False)
    z: int = field(init=False)

    def __post_init__(self) -> None:
        self.dom = self.category.Ty()
        self.z = 0

    def __repr__(self) -> str:
        return f'Word({self.name}, {repr(self.cod), {repr(self.z)}})'

    @property
    def l(self) -> Self:  # noqa: E741, E743
        return type(self)(self.name, self.cod.l)

    @property
    def r(self) -> Self:
        return type(self)(self.name, self.cod.r)

    def dagger(self) -> Daggered | Box:
        return Daggered(self)


Id = Diagram.id
