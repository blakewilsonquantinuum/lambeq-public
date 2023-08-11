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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.
"""
Grammar category
================
Lambeq's internal representation of the grammar category. This work is
based on DisCoPy (https://discopy.org/) which is released under the
BSD 3-Clause "New" or "Revised" License.

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field, InitVar, replace
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

    def __hash__(self) -> int:
        return hash(repr(self))

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

    @overload
    def tensor(self, other: Iterable[Self]) -> Self: ...

    @overload
    def tensor(self, other: Self, *rest: Self) -> Self: ...

    def tensor(self, other: Self | Iterable[Self], *rest: Self) -> Self:
        tys = [*other, *rest]
        if any(not isinstance(ty, type(self)) for ty in tys):
            return NotImplemented

        return self._fromiter(ob for ty in (self, *tys) for ob in ty)

    def __matmul__(self, rhs: Self) -> Self:
        return self.tensor(rhs)

    def rotate(self, z: int) -> Self:
        """Rotate the type, changing the winding number."""
        if self.is_empty or z == 0:
            return self
        elif self.is_atomic:
            return replace(self, z=self.z + z)
        else:
            objects = reversed(self.objects) if z % 2 == 1 else self.objects
            return type(self)(objects=[ob.rotate(z) for ob in objects])

    def unwind(self) -> Self:
        return self.rotate(-self.z)

    @property
    def l(self) -> Self:  # noqa: E741, E743
        return self.rotate(-1)

    @property
    def r(self) -> Self:
        return self.rotate(1)

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
        return type(self)().tensor([self] * times)

    def __pow__(self, times: int) -> Self:
        return self.repeat(times)

    def apply_functor(self, functor: Functor) -> Ty:
        assert not self.is_empty
        if self.is_complex:
            return functor.target_category.Ty().tensor(
                functor(ob) for ob in self.objects
            )
        elif self.z != 0:
            return functor(self.unwind()).rotate(self.z)
        else:
            return functor.ob(self)


class Diagrammable(ABC):
    cod: Ty
    dom: Ty

    @abstractmethod
    def to_diagram(self) -> Diagram: ...

    def __getattr__(self, name: str) -> Any:
        if isinstance(self, Diagram):
            raise AttributeError
        else:
            return getattr(self.to_diagram(), name)

    @abstractmethod
    def apply_functor(self, functor: Functor) -> Diagrammable: ...

    @abstractmethod
    def rotate(self, z: int) -> Diagrammable: ...


@grammar
@dataclass
class Box(Entity, Diagrammable):
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

    def __hash__(self) -> int:
        return hash(repr(self))

    def to_diagram(self) -> Diagram:
        ID = self.category.Ty()
        return self.category.Diagram(dom=self.dom,
                                     cod=self.cod,
                                     layers=[self.category.Layer(box=self,
                                                                 left=ID,
                                                                 right=ID)])

    def __matmul__(self, rhs: Diagrammable) -> Diagram:
        return self.to_diagram().tensor(rhs.to_diagram())

    def __rshift__(self, rhs: Diagrammable) -> Diagram:
        return self.to_diagram().then(rhs.to_diagram())

    def rotate(self, z: int) -> Self:
        """Rotate the box, changing the winding number."""
        return replace(self,
                       dom=self.dom.rotate(z),
                       cod=self.cod.rotate(z),
                       z=self.z + z)

    @property
    def l(self) -> Self:  # noqa: E741, E743
        return self.rotate(-1)

    @property
    def r(self) -> Self:
        return self.rotate(1)

    def unwind(self) -> Self:
        return self.rotate(-self.z)

    def dagger(self) -> Daggered | Box:
        return Daggered(self)

    def apply_functor(self, functor: Functor) -> Diagrammable:
        if self.z != 0:
            return functor(self.unwind()).rotate(self.z)
        else:
            return functor.ar(self)


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
    left: Ty
    box: Box
    right: Ty

    def __repr__(self) -> str:
        return f'|{repr(self.left)} @ {repr(self.box)} @ {repr(self.right)}|'

    def unpack(self) -> tuple[Ty, Box, Ty]:
        return self.left, self.box, self.right

    def extend(self,
               left: Ty | None = None,
               right: Ty | None = None) -> Self:
        ID = self.category.Ty()
        if left is None:
            left = ID
        if right is None:
            right = ID
        return replace(self, left=left @ self.left, right=self.right @ right)

    def rotate(self, z: int) -> Self:
        """Rotate the layer."""
        if z % 2 == 1:
            left, right = self.right, self.left
        else:
            left, right = self.left, self.right

        return replace(self,
                       left=left.rotate(z),
                       box=self.box.rotate(z),
                       right=right.rotate(z))


class InterchangerError(Exception):
    """ This is raised when we try to interchange conected boxes. """
    def __init__(self, box0: Box, box1: Box) -> None:
        super().__init__(f'Boxes {box0} and {box1} do not commute.')


_DiagrammableFactory = Callable[..., Diagrammable]
_DiagrammableFactoryT = TypeVar('_DiagrammableFactoryT',
                                bound=_DiagrammableFactory)


@grammar
@dataclass
class Diagram(Entity, Diagrammable):
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

    special_boxes: ClassVar[dict[str, _DiagrammableFactory]] = {}

    @classmethod
    @overload
    def register_special_box(
        cls,
        name: str,
        diagram_factory: None = None
    ) -> Callable[[_DiagrammableFactoryT], _DiagrammableFactoryT]: ...

    @classmethod
    @overload
    def register_special_box(
        cls,
        name: str,
        diagram_factory: _DiagrammableFactory
    ) -> None: ...

    @classmethod
    def register_special_box(
        cls,
        name: str,
        diagram_factory: _DiagrammableFactory | None = None
    ) -> None | Callable[[_DiagrammableFactoryT], _DiagrammableFactoryT]:
        def set_(
            diagram_factory: _DiagrammableFactoryT
        ) -> _DiagrammableFactoryT:
            cls.special_boxes[name] = diagram_factory
            return diagram_factory

        if diagram_factory is None:
            return set_
        else:
            set_(diagram_factory)
            return None

    def __repr__(self) -> str:
        if self.is_id:
            return f'Id({repr(self.dom)})'
        else:
            return ' >> '.join(map(repr, self.layers))

    def __hash__(self) -> int:
        return hash(repr(self))

    def to_diagram(self) -> Self:
        return self

    @classmethod
    def from_diagrammable(cls, entity: Diagrammable) -> Self:
        return entity.to_diagram()  # type: ignore[return-value]

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

    def __matmul__(self, rhs: Self) -> Self:
        return self.tensor(rhs)

    @property
    def offsets(self) -> list[int]:
        """The offset of a box is the length of the type on its left."""
        return [len(layer.left) for layer in self.layers]

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

    def then_at(self, diagram: Diagrammable, index: int) -> Self:
        return (self
                >> (self.id(self.cod[:index])
                    @ self.from_diagrammable(diagram)
                    @ self.id(self.cod[index+len(diagram.dom):])))

    def __rshift__(self, rhs: Self) -> Self:
        return self.then(rhs)

    def rotate(self, z: int) -> Self:
        """Rotate the diagram."""
        return type(self)(dom=self.dom.rotate(z),
                          cod=self.cod.rotate(z),
                          layers=[layer.rotate(z) for layer in self.layers])

    @property
    def l(self) -> Self:  # noqa: E741, E743
        return self.rotate(-1)

    @property
    def r(self) -> Self:
        return self.rotate(1)

    def dagger(self) -> Self:
        if self.is_id:
            return self
        else:
            return type(self)(dom=self.cod,
                              cod=self.dom,
                              layers=[replace(layer, box=layer.box.dagger())
                                      for layer in reversed(self.layers)])

    @classmethod
    def permutation(cls, dom: Ty, permutation: Iterable[int]) -> Self:
        """Create a layer of Swaps that permutes the wires."""
        permutation = list(permutation)
        if not (len(permutation) == len(dom)
                and set(permutation) == set(range(len(dom)))):
            raise ValueError('Invalid permutation for type of length '
                             f'{len(dom)}: {permutation}')

        wire_index = [*range(len(dom))]

        diagram = cls.id(dom)
        for out_index in range(len(dom) - 1):
            in_index = wire_index[permutation[out_index]]
            assert in_index >= out_index

            for i in reversed(range(out_index, in_index)):
                diagram >>= (
                    cls.id(diagram.cod[:i])
                    @ cls.from_diagrammable(
                        cls.special_boxes['swap'](*diagram.cod[i:i+2])
                    )
                    @ cls.id(diagram.cod[i+2:])
                )

            for i in range(permutation[out_index]):
                wire_index[i] += 1
        return diagram

    def permuted(self, permutation: Iterable[int]) -> Self:
        return self >> self.permutation(self.cod, permutation)

    def interchange(self, i: int, j: int, left: bool = False) -> Diagram:
        """
        Returns a new diagram with boxes i and j interchanged.

        Gets called recursively whenever :code:`i < j + 1 or j < i - 1`.

        Parameters
        ----------
        i : int
            Index of the box to interchange.
        j : int
            Index of the new position for the box.
        left : bool, optional
            Whether to apply left interchangers.

        Notes
        -----
        By default, we apply only right exchange moves::

            top >> Id(left @ box1.dom @ mid) @ box0 @ Id(right)
                >> Id(left) @ box1 @ Id(mid @ box0.cod @ right)
                >> bottom

        gets rewritten to::

            top >> Id(left) @ box1 @ Id(mid @ box0.dom @ right)
                >> Id(left @ box1.cod @ mid) @ box0 @ Id(right)
                >> bottom
        """
        if not 0 <= i < len(self) or not 0 <= j < len(self):
            raise IndexError
        if i == j:
            return self
        if j < i - 1:
            result = self
            for k in range(i - j):
                result = result.interchange(i - k, i - k - 1, left=left)
            return result
        if j > i + 1:
            result = self
            for k in range(j - i):
                result = result.interchange(i + k, i + k + 1, left=left)
            return result
        if j < i:
            i, j = j, i
        off0, off1 = self.offsets[i], self.offsets[j]
        left0, box0, right0 = self.layers[i].unpack()
        left1, box1, right1 = self.layers[j].unpack()
        # By default, we check if box0 is to the right first,
        # then to the left.
        if left and off1 >= off0 + len(box0.cod):  # box0 left of box1
            middle = left1[len(left0 @ box0.cod):]
            layer0 = self.category.Layer(left0, box0,
                                         middle @ box1.cod @ right1)
            layer1 = self.category.Layer(left0 @ box0.dom @ middle, box1,
                                         right1)
        elif off0 >= off1 + len(box1.dom):  # box0 right of box1
            middle = left0[len(left1 @ box1.dom):]
            layer0 = self.category.Layer(left1 @ box1.cod @ middle, box0,
                                         right0)
            layer1 = self.category.Layer(left1, box1,
                                         middle @ box0.dom @ right0)
        elif off1 >= off0 + len(box0.cod):  # box0 left of box1
            middle = left1[len(left0 @ box0.cod):]
            layer0 = self.category.Layer(left0, box0,
                                         middle @ box1.cod @ right1)
            layer1 = self.category.Layer(left0 @ box0.dom @ middle, box1,
                                         right1)
        else:
            raise InterchangerError(box0, box1)
        layers = self.layers[:i] + [layer1, layer0] + self.layers[i + 2:]
        return Diagram(self.dom, self.cod, layers=layers)

    def normalize(self, left: bool = False) -> Iterator[Diagram]:
        """
        Implements normalization of diagrams,
        see arXiv:1804.07832.

        Parameters
        ----------
        left : bool, optional
            Passed to :meth:`Diagram.interchange`.

        Yields
        ------
        diagram : :class:`Diagram`
            Rewrite steps.

        Examples
        --------
        >>> s0, s1 = Box('s0', Ty(), Ty()), Box('s1', Ty(), Ty())
        >>> gen = (s0 @ s1).normalize()
        >>> for _ in range(3): print(next(gen))
        |Ty() @ [s1; Ty() -> Ty()] @ Ty()| >> \
|Ty() @ [s0; Ty() -> Ty()] @ Ty()|
        |Ty() @ [s0; Ty() -> Ty()] @ Ty()| >> \
|Ty() @ [s1; Ty() -> Ty()] @ Ty()|
        |Ty() @ [s1; Ty() -> Ty()] @ Ty()| >> \
|Ty() @ [s0; Ty() -> Ty()] @ Ty()|
        """
        diagram = self
        no_more_moves = False
        while not no_more_moves:
            no_more_moves = True
            for i in range(len(diagram) - 1):
                box0, box1 = diagram.boxes[i], diagram.boxes[i + 1]
                off0, off1 = diagram.offsets[i], diagram.offsets[i + 1]
                if ((left and off1 >= off0 + len(box0.cod))
                        or (not left and off0 >= off1 + len(box1.dom))):
                    diagram = diagram.interchange(i, i + 1, left=left)
                    yield diagram
                    no_more_moves = False

    def normal_form(self, left: bool = False) -> Diagram:
        """
        Returns the normal form of a connected diagram,
        see arXiv:1804.07832.

        Parameters
        ----------
        left : bool, optional
            Whether to apply left interchangers.

        Raises
        ------
        NotImplementedError
            Whenever :code:`normalizer` yields the same rewrite steps
            twice.
        """
        diagram, cache = self, set()
        for _diagram in diagram.snake_removal(left=left):
            if _diagram in cache:
                raise NotImplementedError(f'{str(self)} is not connected.')
            diagram = _diagram
            cache.add(diagram)
        return diagram

    def snake_removal(self, left: bool = False) -> Iterator[Diagram]:
        """
        Returns a generator which yields normalization steps.

        Parameters
        ----------
        left : bool, optional
            Whether to apply left interchangers.

        Yields
        ------
        diagram : :class:`Diagram`
            Rewrite steps.

        Examples
        --------
        >>> n, s = Ty('n'), Ty('s')
        >>> cup, cap = Cup(n, n.r), Cap(n.r, n)
        >>> f = Box('f', n, n)
        >>> g = Box('g', s @ n, n)
        >>> h = Box('h', n, n @ s)
        >>> diagram = g @ cap >> f.dagger() @ Id(n.r) @ f >> cup @ h
        >>> for d in diagram.snake_removal():
        ...     print(d)  # doctest: +ELLIPSIS
        |Ty... >> |Ty() @ [CUP; Ty(n) @ Ty(n).r -> Ty()] @ Ty(n)| >>...
        |Ty... >> |Ty(n) @ [CAP; Ty() -> Ty(n).r @ Ty(n)] @ Ty()| >> \
|Ty() @ [CUP; Ty(n) @ Ty(n).r -> Ty()] @ Ty(n)| >>...
        |Ty() @ [g; Ty(s) @ Ty(n) -> Ty(n)] @ Ty()| >> \
|Ty() @ [f†; Ty(n) -> Ty(n)] @ Ty()| >> \
|Ty() @ [f; Ty(n) -> Ty(n)] @ Ty()| >> \
|Ty() @ [h; Ty(n) -> Ty(n) @ Ty(s)] @ Ty()|
        """
        def follow_wire(diagram: Diagram,
                        i: int,
                        j: int) -> tuple[int, int,
                                         tuple[list[int], list[int]]]:
            """
            Given a diagram, the index of a box i and the offset j of an
            output wire, returns (i, j, obstructions) where:
            - i is the index of the box which takes this wire as input,
            or len(diagram) if it is connected to the bottom boundary.
            - j is the offset of the wire at its bottom end.
            - obstructions is a pair of lists of indices for the boxes
            on the left and right of the wire we followed.
            """
            left_obstruction = []  # type: list[int]
            right_obstruction = []  # type: list[int]
            while i < len(diagram) - 1:
                i += 1
                box, off = diagram.boxes[i], diagram.offsets[i]
                if off <= j < off + len(box.dom):
                    return i, j, (left_obstruction, right_obstruction)
                if off <= j:
                    j += len(box.cod) - len(box.dom)
                    left_obstruction.append(i)
                else:
                    right_obstruction.append(i)
            return len(diagram), j, (left_obstruction, right_obstruction)

        def find_snake(diagram: Diagram) -> None | tuple[int, int,
                                                         tuple[list[int],
                                                               list[int]],
                                                         bool]:
            """
            Given a diagram, returns (cup, cap, obstructions,
            left_snake) if there is a yankable pair, otherwise returns
            None.
            """
            for cap in range(len(diagram)):
                if not isinstance(diagram.boxes[cap], Cap):
                    continue
                for left_snake, wire in [(True, diagram.offsets[cap]),
                                         (False, diagram.offsets[cap] + 1)]:
                    cup, wire, obstructions = follow_wire(diagram, cap, wire)
                    not_yankable = (cup == len(diagram)
                                    or not isinstance(diagram.boxes[cup], Cup)
                                    or (left_snake
                                        and diagram.offsets[cup] + 1 != wire)
                                    or (not left_snake
                                        and diagram.offsets[cup] != wire))
                    if not_yankable:
                        continue
                    return cup, cap, obstructions, left_snake
            return None

        def unsnake(diagram: Diagram,
                    cup: int,
                    cap: int,
                    obstructions: tuple[list[int], list[int]],
                    left_snake: bool = False) -> Iterator[Diagram]:
            """
            Given a diagram and the indices for a cup and cap pair
            and a pair of lists of obstructions on the left and right,
            returns a new diagram with the snake removed.

            A left snake is one of the form Id @ Cap >> Cup @ Id.
            A right snake is one of the form Cap @ Id >> Id @ Cup.
            """
            left_obstruction, right_obstruction = obstructions
            if left_snake:
                for box in left_obstruction:
                    diagram = diagram.interchange(box, cap)
                    yield diagram
                    for i, right_box in enumerate(right_obstruction):
                        if right_box < box:
                            right_obstruction[i] += 1
                    cap += 1
                for box in right_obstruction[::-1]:
                    diagram = diagram.interchange(box, cup)
                    yield diagram
                    cup -= 1
            else:
                for box in left_obstruction[::-1]:
                    diagram = diagram.interchange(box, cup)
                    yield diagram
                    for i, right_box in enumerate(right_obstruction):
                        if right_box > box:
                            right_obstruction[i] -= 1
                    cup -= 1
                for box in right_obstruction:
                    diagram = diagram.interchange(box, cap)
                    yield diagram
                    cap += 1
            layers = diagram.layers[:cap] + diagram.layers[cup + 1:]
            yield Diagram(diagram.dom, diagram.cod, layers)

        diagram = self
        while True:
            yankable = find_snake(diagram)
            if yankable is None:
                break
            for _diagram in unsnake(diagram, *yankable):
                yield _diagram
                diagram = _diagram
        for _diagram in diagram.normalize(left=left):
            yield _diagram

    def draw(self, **kwargs: Any) -> None:
        from lambeq.backend.drawing import draw
        draw(self, **kwargs)

    def apply_functor(self, functor: Functor) -> Diagram:
        assert not self.is_id
        diagram = functor(self.id(self.dom))
        for layer in self.layers:
            left, box, right = layer.unpack()
            diagram >>= (functor(self.id(left))
                         @ functor(box).to_diagram()
                         @ functor(self.id(right)))
        return diagram


@Diagram.register_special_box('cap')
@dataclass
class Cap(Box):
    """The unit of the adjunction for an atomic type.

    Parameters
    ----------
    left : Ty
        The type of the left output.
    right : Ty
        The type of the right output.
    is_reversed : bool, default: False
        Whether the cap is reversed or not. Normally, caps only allow
        outputs where `right` is the left adjoint of `left`. However,
        to facilitate operations like `dagger`, we pass in a flag that
        indicates that the inputs are the opposite way round, which
        initialises a reversed cap. Then, when a cap is adjointed, it
        turns into a reversed cap, which can be adjointed again to turn
        it back into a normal cap.

    """
    left: Ty
    right: Ty
    is_reversed: InitVar[bool] = False

    name: str = field(init=False)
    dom: Ty = field(init=False)
    cod: Ty = field(init=False)
    z: int = field(init=False)

    def __post_init__(self, is_reversed: bool) -> None:
        if not self.left.is_atomic or not self.right.is_atomic:
            raise ValueError('left and right need to be atomic types.')
        self._check_adjoint(self.left, self.right, is_reversed)

        self.name = 'CAP'
        self.dom = self.category.Ty()
        self.cod = self.left @ self.right
        self.z = int(is_reversed)

    @staticmethod
    def _check_adjoint(left: Ty, right: Ty, is_reversed: bool) -> None:
        if is_reversed:
            if left != right.l:
                raise ValueError('left and right need to be adjoints')
        else:
            if left != right.r:
                raise ValueError('left and right need to be adjoints')

    def __new__(cls,  # type: ignore[misc]
                left: Ty,
                right: Ty,
                is_reversed: bool = False) -> Diagrammable:
        if left.is_atomic and right.is_atomic:
            return super().__new__(cls)
        else:
            cls._check_adjoint(left, right, is_reversed)

            diagram = cls.category.Diagram.id()
            for i, (l_ob, r_ob) in enumerate(zip(left, reversed(right))):
                diagram = diagram.then_at(cls(l_ob, r_ob), i)
            return diagram

    @classmethod
    def to_right(cls, left: Ty, is_reversed: bool = False) -> Self | Diagram:
        return cls(left, left.r if is_reversed else left.l)

    @classmethod
    def to_left(cls, right: Ty, is_reversed: bool = False) -> Self | Diagram:
        return cls(right.l if is_reversed else right.r, right)

    def rotate(self, z: int) -> Self:
        """Rotate the cap."""
        if z % 2 == 1:
            left, right = self.right, self.left
        else:
            left, right = self.left, self.right
        is_reversed = (self.z + z) % 2 == 1
        return type(self)(left.rotate(z),
                          right.rotate(z),
                          is_reversed=is_reversed)

    def dagger(self) -> Cup:
        Cup = self.category.Diagram.special_boxes['cup']
        return Cup(self.left,  # type: ignore[return-value]
                   self.right,
                   is_reversed=not self.z)

    def apply_functor(self, functor: Functor) -> Diagrammable:
        return functor.target_category.Diagram.special_boxes['cap'](
            functor(self.left),
            functor(self.right),
            is_reversed=bool(self.z)
        )

    __repr__ = Box.__repr__
    __hash__ = Box.__hash__


@Diagram.register_special_box('cup')
@dataclass
class Cup(Box):
    """The counit of the adjunction for an atomic type.

    Parameters
    ----------
    left : Ty
        The type of the left output.
    right : Ty
        The type of the right output.
    is_reversed : bool, default: False
        Whether the cup is reversed or not. Normally, cups only allow
        inputs where `right` is the right adjoint of `left`. However,
        to facilitate operations like `dagger`, we pass in a flag that
        indicates that the inputs are the opposite way round, which
        initialises a reversed cup. Then, when a cup is adjointed, it
        turns into a reversed cup, which can be adjointed again to turn
        it back into a normal cup.

    """
    left: Ty
    right: Ty
    is_reversed: InitVar[bool] = False

    name: str = field(init=False)
    dom: Ty = field(init=False)
    cod: Ty = field(init=False)
    z: int = field(init=False)

    def __post_init__(self, is_reversed: bool) -> None:
        if not self.left.is_atomic or not self.right.is_atomic:
            raise ValueError('left and right need to be atomic types.')
        self._check_adjoint(self.left, self.right, is_reversed)

        self.name = 'CUP'
        self.dom = self.left @ self.right
        self.cod = self.category.Ty()
        self.z = int(is_reversed)

    @staticmethod
    def _check_adjoint(left: Ty, right: Ty, is_reversed: bool) -> None:
        if is_reversed:
            if left != right.r:
                raise ValueError('left and right need to be adjoints')
        else:
            if left != right.l:
                raise ValueError('left and right need to be adjoints')

    def __new__(cls,  # type: ignore[misc]
                left: Ty,
                right: Ty,
                is_reversed: bool = False) -> Diagrammable:
        if left.is_atomic and right.is_atomic:
            return super().__new__(cls)
        else:
            cls._check_adjoint(left, right, is_reversed)

            diagram = cls.category.Diagram.id(left @ right)
            for i, (l_ob, r_ob) in enumerate(zip(reversed(left), right)):
                diagram = diagram.then_at(cls(l_ob, r_ob), len(left) - 1 - i)
            return diagram

    @classmethod
    def to_right(cls, left: Ty, is_reversed: bool = False) -> Self | Diagram:
        return cls(left, left.l if is_reversed else left.r)

    @classmethod
    def to_left(cls, right: Ty, is_reversed: bool = False) -> Self | Diagram:
        return cls(right.r if is_reversed else right.l, right)

    def rotate(self, z: int) -> Self:
        """Rotate the cup."""
        if z % 2 == 1:
            left, right = self.right, self.left
        else:
            left, right = self.left, self.right
        is_reversed = (self.z + z) % 2 == 1
        return type(self)(left.rotate(z),
                          right.rotate(z),
                          is_reversed=is_reversed)

    def dagger(self) -> Cap:
        Cap = self.category.Diagram.special_boxes['cap']
        return Cap(  # type: ignore[return-value]
            self.left,
            self.right,
            is_reversed=not self.z
        )

    def apply_functor(self, functor: Functor) -> Diagrammable:
        return functor.target_category.Diagram.special_boxes['cup'](
            functor(self.left),
            functor(self.right),
            is_reversed=bool(self.z)
        )

    __repr__ = Box.__repr__
    __hash__ = Box.__hash__


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

    def rotate(self, z: int) -> Self:
        """Rotate the daggered box."""
        return type(self)(self.box.rotate(z))

    def dagger(self) -> Box:
        return self.box

    __repr__ = Box.__repr__
    __hash__ = Box.__hash__


@Diagram.register_special_box('spider')
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
    z: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if not self.type.is_atomic:
            raise TypeError('Spider type needs to be atomic.')
        self.name = 'SPIDER'
        self.dom = self.type ** self.n_legs_in
        self.cod = self.type ** self.n_legs_out

    def __new__(cls,  # type: ignore[misc]
                type: Ty,
                n_legs_in: int,
                n_legs_out: int) -> Diagrammable:
        if type.is_atomic:
            return super().__new__(cls)
        else:
            size = len(type)
            total_legs_in = size * n_legs_in
            return (
                cls.category.Diagram.permutation(
                    type ** n_legs_in,
                    [j
                     for i in range(size)
                     for j in range(i, total_legs_in, size)]
                )
                >> cls.category.Diagram.id().tensor(
                    *(cls(ob, n_legs_in, n_legs_out)
                      for ob in type)  # type: ignore[arg-type]
                ).permuted([
                    j
                    for i in range(n_legs_out)
                    for j in range(i, len(type) * n_legs_out, n_legs_out)
                ])
            )

    def rotate(self, z: int) -> Self:
        """Rotate the spider."""
        return type(self)(self.type.rotate(z), len(self.dom), len(self.cod))

    def dagger(self) -> Self:
        return type(self)(self.type, self.n_legs_out, self.n_legs_in)

    def apply_functor(self, functor: Functor) -> Diagrammable:
        return functor.target_category.Diagram.special_boxes['spider'](
            functor(self.type),
            self.n_legs_in,
            self.n_legs_out
        )

    __repr__ = Box.__repr__
    __hash__ = Box.__hash__


@Diagram.register_special_box('swap')
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
    z: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if not self.left.is_atomic or not self.right.is_atomic:
            raise ValueError('Types need to be atomic')

        self.name = 'SWAP'
        self.dom = self.left @ self.right
        self.cod = self.right @ self.left

    def __new__(cls,  # type: ignore[misc]
                left: Ty,
                right: Ty) -> Swap | Diagram:
        if left.is_atomic and right.is_atomic:
            return super().__new__(cls)
        else:
            diagram = cls.category.Diagram.id(left @ right)
            for start, ob in enumerate(right):
                for i in reversed(range(len(left))):
                    diagram = diagram.then_at(cls(left[i], ob), start + i)
            return diagram

    def rotate(self, z: int) -> Self:
        """Rotate the swap."""
        if z % 2 == 1:
            left, right = self.right, self.left
        else:
            left, right = self.left, self.right
        return type(self)(left.rotate(z), right.rotate(z))

    def dagger(self) -> Self:
        return type(self)(self.right, self.left)

    def apply_functor(self, functor: Functor) -> Diagrammable:
        return functor.target_category.Diagram.special_boxes['swap'](
            functor(self.left),
            functor(self.right)
        )

    __repr__ = Box.__repr__
    __hash__ = Box.__hash__


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
    z: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.dom = self.category.Ty()

    def __repr__(self) -> str:
        return f'Word({self.name}, {repr(self.cod), {repr(self.z)}})'

    def __hash__(self) -> int:
        return hash(repr(self))

    def rotate(self, z: int) -> Self:
        """Rotate the Word box, changing the winding number."""
        return type(self)(self.name, self.cod.rotate(z))

    def dagger(self) -> Daggered:
        return Daggered(self)


Id = Diagram.id


@dataclass(init=False)
class Functor:
    """A functor that maps between categories.

    Parameters
    ----------
    target_category : Category
        The category to which the functor maps.
    ob : callable, optional
        A function that maps types to types, by default None
    ar : callable, optional
        A function that maps boxes to Diagrammables, by default None

    Examples
    --------

    >>> n = Ty('n')
    >>> diag = Cap(n, n.l) @ Id(n) >> Id(n) @ Cup(n.l, n)
    >>> diag.draw(
    ...     figsize=(2, 2), path='./docs/_static/images/snake.png')

    .. image:: ./docs/_static/images/snake.png
        :align: center

    >>> F = Functor(grammar, lambda _, ty : ty @ ty)
    >>> F(diag).draw(
    ...     figsize=(2, 2), path='./docs/_static/images/snake-2.png')

    .. image:: ./docs/_static/images/snake-2.png
        :align: center

    """
    target_category: Category

    def __init__(
        self,
        target_category: Category,
        ob: Callable[[Functor, Ty], Ty],
        ar: Callable[[Functor, Box], Diagrammable] | None = None
    ) -> None:
        self.target_category = target_category
        self.custom_ob = ob
        self.custom_ar = ar
        self.ob_cache: dict[Ty, Ty] = {}
        self.ar_cache: dict[Diagrammable, Diagrammable] = {}

    @overload
    def __call__(self, entity: Ty) -> Ty: ...

    @overload
    def __call__(self, entity: Box) -> Diagrammable: ...

    @overload
    def __call__(self, entity: Diagram) -> Diagram: ...

    def __call__(self, entity: Ty | Diagrammable) -> Ty | Diagrammable:
        """Apply the functor to a type or a box.

        Parameters
        ----------
        entity : Ty or Diagrammable
            The type or box to which the functor is applied.

        """
        if isinstance(entity, Ty):
            return self.ob_with_cache(entity)
        else:
            return self.ar_with_cache(entity)

    def ob_with_cache(self, ob: Ty) -> Ty:
        """Apply the functor to a type, caching the result."""
        try:
            return self.ob_cache[ob]
        except KeyError:
            pass

        if ob.is_empty:
            ret = self.target_category.Ty()
        else:
            ret = ob.apply_functor(self)

        self.ob_cache[ob] = ret
        return ret

    def ar_with_cache(self, ar: Diagrammable) -> Diagrammable:
        """Apply the functor to a Diagrammable, caching the result."""
        try:
            return self.ar_cache[ar]
        except KeyError:
            pass

        if not ar.is_id:
            ret = ar.apply_functor(self)
        else:
            ret = self.target_category.Diagram.id(self.ob_with_cache(ar.dom))

        self.ar_cache[ar] = ret

        cod_check = self.ob_with_cache(ar.cod)
        dom_check = self.ob_with_cache(ar.dom)
        if ret.cod != cod_check or ret.dom != dom_check:
            raise TypeError(f'The arrow is ill-defined. Applying the functor '
                            f'to a box returns dom = {ret.dom}, cod = '
                            f'{ret.cod} expected dom = {dom_check}, cod = '
                            f'{cod_check}')
        return ret

    def ob(self, ob: Ty) -> Ty:
        """Apply the functor to a type."""
        if self.custom_ob is None:
            raise AttributeError('Specify a custom ob function if you want to '
                                 'use the functor on types.')
        return self.custom_ob(self, ob)

    def ar(self, ar: Box) -> Diagrammable:
        """Apply the functor to a box."""
        if self.custom_ar is None:
            raise AttributeError('Specify a custom ar function if you want to '
                                 'use the functor on boxes.')

        return self.custom_ar(self, ar)
