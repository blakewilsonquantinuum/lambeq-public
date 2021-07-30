from __future__ import annotations

__all__ = ['CCGTree']

import json
from typing import Any, Dict, Optional, Sequence, Tuple, Union, overload

from discopy import biclosed, rigid, Word
from discopy.biclosed import Diagram, Functor, Id, Ty, biclosed2rigid_ob

from discoket.ccg2discocat.ccg_rule import CCGRule, GBC, GBX, GFC, GFX
from discoket.ccg2discocat.ccg_types import (CCGAtomicType, replace_cat_result,
                                             str2biclosed)

# Types
_JSONDictT = Dict[str, Any]


class PlanarBX(biclosed.Box):
    """Planar Backward Crossed Composition Box."""
    def __init__(self, dom: Ty, diagram: Diagram) -> None:
        assert isinstance(dom, biclosed.Over)
        assert not diagram.dom

        right = diagram.cod
        assert isinstance(right, biclosed.Under)
        assert right.left == dom.left

        self.diagram = diagram
        cod = right.right << dom.right
        super().__init__(f'PlanarBX({dom}, {diagram})', dom, cod)


class PlanarFX(biclosed.Box):
    """Planar Forward Crossed Composition Box."""
    def __init__(self, dom: Ty, diagram: Diagram) -> None:
        assert isinstance(dom, biclosed.Under)
        assert not diagram.dom

        left = diagram.cod
        assert isinstance(left, biclosed.Over)
        assert left.right == dom.right

        self.diagram = diagram
        cod = dom.left >> left.left
        super().__init__(f'PlanarFX({dom}, {diagram})', dom, cod)


class PlanarGBX(biclosed.Box):
    def __init__(self, dom: Ty, diagram: Diagram):
        assert not diagram.dom

        right = diagram.cod
        assert isinstance(right, biclosed.Under)

        cod, original = replace_cat_result(dom, right.left, right.right, '/|')
        assert original == right.left

        self.diagram = diagram
        super().__init__(f'PlanarGBX({dom}, {diagram})', dom, cod)


class PlanarGFX(biclosed.Box):
    def __init__(self, dom: Ty, diagram: Diagram):
        assert not diagram.dom

        left = diagram.cod
        assert isinstance(left, biclosed.Over)

        cod, original = replace_cat_result(dom, left.right, left.left, r'\|')
        assert original == left.right

        self.diagram = diagram
        super().__init__(f'PlanarGFX({dom}, {diagram})', dom, cod)


def biclosed2str(biclosed_type: Ty, pretty: bool = False) -> str:
    if isinstance(biclosed_type, biclosed.Over):
        template = '({0}↢{1})' if pretty else '({0}/{1})'
    elif isinstance(biclosed_type, biclosed.Under):
        template = '({0}↣{1})' if pretty else r'({1}\{0})'
    else:
        return str(biclosed_type)
    return template.format(biclosed2str(biclosed_type.left, pretty),
                           biclosed2str(biclosed_type.right, pretty))


class CCGTree:
    """Derivation tree for a CCG.

    This provides a standard derivation interface between the parser and
    the rest of the model.
    """

    def __init__(self,
                 text: Optional[str] = None,
                 *,
                 rule: Union[str, CCGRule] = CCGRule.UNKNOWN,
                 biclosed_type: Ty,
                 children: Optional[Sequence[CCGTree]] = None) -> None:
        self._text = text
        self.rule = CCGRule(rule)
        self.biclosed_type = biclosed_type
        self.children = children if children is not None else []

        n_children = len(self.children)
        child_requirements = {CCGRule.LEXICAL: 0,
                              CCGRule.UNARY: 1,
                              CCGRule.FORWARD_TYPE_RAISING: 1,
                              CCGRule.BACKWARD_TYPE_RAISING: 1}
        if (self.rule != CCGRule.UNKNOWN and
                child_requirements.get(self.rule, 2) != n_children):
            raise ValueError(f'Invalid number of children ({n_children}) for '
                             f'rule "{self.rule}"')

        if text and not children:
            self.rule = CCGRule.LEXICAL

    @property
    def text(self) -> str:
        if self._text is None:
            self._text = ' '.join(child.text for child in self.children)
        return self._text

    @overload
    @classmethod
    def from_json(cls, data: None) -> None: ...

    @overload
    @classmethod
    def from_json(cls, data: Union[str, _JSONDictT]) -> CCGTree: ...

    @classmethod
    def from_json(cls,
                  data: Union[None, str, _JSONDictT]) -> Optional[CCGTree]:
        if data is None:
            return None

        data_dict = json.loads(data) if isinstance(data, str) else data
        return cls(text=data_dict.get('text'),
                   rule=data_dict.get('rule', CCGRule.UNKNOWN),
                   biclosed_type=str2biclosed(data_dict['type']),
                   children=[cls.from_json(child)
                             for child in data_dict.get('children', [])])

    def to_json(self) -> _JSONDictT:
        data: _JSONDictT = {'type': biclosed2str(self.biclosed_type)}
        if self.rule != CCGRule.UNKNOWN:
            data['rule'] = self.rule.value
        if self.text != ' '.join(child.text for child in self.children):
            data['text'] = self.text
        if self.children:
            data['children'] = [child.to_json() for child in self.children]
        return data

    def __eq__(self, other: Any) -> bool:
        return (isinstance(other, CCGTree) and
                self.text == other.text and
                self.rule == other.rule and
                self.biclosed_type == other.biclosed_type and
                len(self.children) == len(other.children) and
                all(c1 == c2 for c1, c2 in zip(self.children, other.children)))

    def __repr__(self) -> str:  # pragma: no cover
        return f'{type(self).__name__}("{self.text}")'

    def deriv(self,
              use_slashes: bool = False,
              prefix: str = '') -> str:  # pragma: no cover
        output_type = biclosed2str(self.biclosed_type, not use_slashes)
        if self.rule == CCGRule.LEXICAL:
            deriv = f' {output_type} ∋ "{self.text}"'
        else:
            deriv = (f'{self.rule}: {output_type} ← ' +
                     ' + '.join(biclosed2str(child.biclosed_type, True)
                                for child in self.children))
        deriv = f'{prefix}{deriv}'

        if self.children:
            if prefix:
                prefix = prefix[:-1] + ('│ ' if prefix[-1] == '├' else '  ')
            for child in self.children[:-1]:
                deriv += '\n' + child.deriv(use_slashes, prefix + '├')
            deriv += '\n' + self.children[-1].deriv(use_slashes, prefix + '└')
        return deriv

    def to_biclosed_diagram(self,
                            separate_words: bool = False,
                            planar: bool = False,
                            resolved_output: Optional[Ty] = None) -> Diagram:
        biclosed_type = resolved_output or self.biclosed_type

        if self.rule == CCGRule.LEXICAL:
            word = biclosed.Box(self.text, Ty(), biclosed_type)
            return (word, Id(biclosed_type)) if separate_words else word

        child_types = [child.biclosed_type for child in self.children]

        this_layer = self.rule(Ty.tensor(*child_types), biclosed_type)

        children = [child.to_biclosed_diagram(True,
                                              planar,
                                              this_layer.dom[i:i+1])
                    for i, child in enumerate(self.children)]

        if planar and self.rule == CCGRule.BACKWARD_CROSSED_COMPOSITION:
            (words, left_diag), (right_words, right_diag) = children
            diag = (left_diag >>
                    PlanarBX(left_diag.cod, right_words >> right_diag))
        elif planar and self.rule == CCGRule.FORWARD_CROSSED_COMPOSITION:
            (left_words, left_diag), (words, right_diag) = children
            diag = (right_diag >>
                    PlanarFX(right_diag.cod, left_words >> left_diag))
        elif (planar and
              self.rule == CCGRule.GENERALIZED_BACKWARD_CROSSED_COMPOSITION):
            (words, left_diag), (right_words, right_diag) = children
            diag = (left_diag >>
                    PlanarGBX(left_diag.cod, right_words >> right_diag))
        elif (planar and
              self.rule == CCGRule.GENERALIZED_FORWARD_CROSSED_COMPOSITION):
            (left_words, left_diag), (words, right_diag) = children
            diag = (right_diag >>
                    PlanarGFX(right_diag.cod, left_words >> left_diag))
        else:
            words, diag = [Diagram.tensor(*d) for d in zip(*children)]
            diag >>= this_layer

        if separate_words:
            return words, diag
        else:
            return words >> diag

    def to_diagram(self, planar: bool = False) -> rigid.Diagram:
        def ob_func(ob: Ty) -> rigid.Ty:
            return (rigid.Ty() if ob == CCGAtomicType.PUNCTUATION
                    else biclosed2rigid_ob(ob))

        def ar_func(box: biclosed.Box) -> rigid.Diagram:
            #           special box -> special diagram
            # RemovePunctuation box -> identity wire(s)
            #           punctuation -> empty diagram
            #              word box -> Word

            def split(cat: Ty,
                      base: Ty) -> Tuple[rigid.Ty, rigid.Ty, rigid.Ty]:
                left = right = rigid.Ty()
                while cat != base:
                    if isinstance(cat, biclosed.Over):
                        right = to_rigid_diagram(cat.right).l @ right
                        cat = cat.left
                    else:
                        left @= to_rigid_diagram(cat.left).r
                        cat = cat.right
                return left, to_rigid_diagram(cat), right

            if isinstance(box, PlanarBX):
                join = to_rigid_diagram(box.dom.left)
                right = to_rigid_diagram(box.dom)[len(join):]
                inner = to_rigid_diagram(box.diagram)
                cups = rigid.cups(join, join.r)
                return (Id(join) @ inner >>
                        cups @ Id(inner.cod[len(join):])) @ Id(right)

            if isinstance(box, PlanarFX):
                join = to_rigid_diagram(box.dom.right)
                left = to_rigid_diagram(box.dom)[:-len(join)]
                inner = to_rigid_diagram(box.diagram)
                cups = rigid.cups(join.l, join)
                return Id(left) @ (inner @ Id(join) >>
                                   Id(inner.cod[:-len(join)]) @ cups)

            if isinstance(box, PlanarGBX):
                left, join, right = split(box.dom, box.diagram.cod.left)
                inner = to_rigid_diagram(box.diagram)
                cups = rigid.cups(join, join.r)
                mid = (Id(join) @ inner) >> (cups @ Id(inner.cod[len(join):]))
                return Id(left) @ mid @ Id(right)

            if isinstance(box, PlanarGFX):
                left, join, right = split(box.dom, box.diagram.cod.right)
                inner = to_rigid_diagram(box.diagram)
                cups = rigid.cups(join.l, join)
                mid = (inner @ Id(join)) >> (Id(inner.cod[:-len(join)]) @ cups)
                return Id(left) @ mid @ Id(right)

            if isinstance(box, GBC):
                left = to_rigid_diagram(box.dom[0])
                mid = to_rigid_diagram(box.dom[1].left)
                right = to_rigid_diagram(box.dom[1].right)
                return (Id(left[:-len(mid)]) @ rigid.cups(mid, mid.r) @
                        Id(right))

            if isinstance(box, GFC):
                left = to_rigid_diagram(box.dom[0].left)
                mid = to_rigid_diagram(box.dom[0].right)
                right = to_rigid_diagram(box.dom[1])
                return Id(left) @ rigid.cups(mid.l, mid) @ Id(right[len(mid):])

            if isinstance(box, GBX):
                mid = to_rigid_diagram(box.dom[1].right)
                left, join, right = split(box.dom[0], box.dom[1].left)
                swaps = rigid.Diagram.swap(right, join >> mid)
                return Id(left) @ (Id(join) @ swaps >>
                                   rigid.cups(join, join.r) @ Id(mid @ right))

            if isinstance(box, GFX):
                mid = to_rigid_diagram(box.dom[0].left)
                left, join, right = split(box.dom[1], box.dom[0].right)
                return (rigid.Diagram.swap(mid << join, left) @ Id(join) >>
                        Id(left @ mid) @ rigid.cups(join.l, join)) @ Id(right)

            cod = to_rigid_diagram(box.cod)
            return Id(cod) if box.dom or not cod else Word(box.name, cod)

        to_rigid_diagram = Functor(ob=ob_func,
                                   ar=ar_func,
                                   ob_factory=rigid.Ty,
                                   ar_factory=rigid.Diagram)
        return to_rigid_diagram(self.to_biclosed_diagram(planar=planar))
