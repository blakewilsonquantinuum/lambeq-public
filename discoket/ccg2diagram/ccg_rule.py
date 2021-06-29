__all__ = ['CCGAtomicType', 'CCGRule', 'CCGRuleUseError']

from enum import Enum
from typing import Any

from discopy import rigid
from discopy.biclosed import Box, Diagram, Id, Ty

from discoket.core.types import AtomicType


class CCGRuleUseError(Exception):
    def __init__(self, rule: 'CCGRule', message: str) -> None:
        self.rule = rule
        self.message = message

    def __str__(self) -> str:
        return f'Illegal use of {self.rule}: {self.message}'


class CCGAtomicType(Ty, Enum):
    """Standard CCG atomic types mapping to their biclosed type."""

    def __new__(cls, value: rigid.Ty) -> Ty:
        return Ty(value[0])

    NOUN = AtomicType.NOUN
    NOUN_PHRASE = AtomicType.NOUN_PHRASE
    SENTENCE = AtomicType.SENTENCE
    PREPOSITION = AtomicType.PREPOSITION
    CONJUNCTION = AtomicType.CONJUNCTION
    PUNCTUATION = AtomicType.PUNCTUATION

    @classmethod
    def conjoinable(cls, _type: Any) -> bool:
        return _type in (cls.CONJUNCTION, cls.PUNCTUATION)


class RPL(Box):
    """Remove left punctuation box."""
    def __init__(self, left: Ty, right: Ty) -> None:
        dom, cod = left @ right, right
        super().__init__(f'RPL({left}, {right})', dom, cod)


class RPR(Box):
    """Remove right punctuation box."""
    def __init__(self, left: Ty, right: Ty) -> None:
        dom, cod = left @ right, left
        super().__init__(f'RPR({left}, {right})', dom, cod)


class CCGRule(str, Enum):
    FORWARD_APPLICATION = 'FA'
    BACKWARD_APPLICATION = 'BA'
    FORWARD_COMPOSITION = 'FC'
    BACKWARD_COMPOSITION = 'BC'
    FORWARD_CROSSED_COMPOSITION = 'FX'
    BACKWARD_CROSSED_COMPOSITION = 'BX'
    REMOVE_PUNCTUATION_LEFT = 'LP'
    REMOVE_PUNCTUATION_RIGHT = 'RP'
    FORWARD_TYPE_RAISING = 'FTR'
    BACKWARD_TYPE_RAISING = 'BTR'
    CONJUNCTION = 'CONJ'
    LEXICAL = 'LEX'
    UNKNOWN = 'UNK'

    @classmethod
    def _missing_(cls, _: Any) -> 'CCGRule':
        return cls.UNKNOWN

    def __call__(self, input_type: Ty, output_type: Ty) -> Diagram:
        if self == self.FORWARD_APPLICATION:
            return Diagram.fa(output_type, input_type[1:])
        elif self == self.BACKWARD_APPLICATION:
            return Diagram.ba(input_type[:1], output_type)
        elif self == self.FORWARD_COMPOSITION:
            assert input_type[0].right == input_type[1].left
            l, m, r = output_type.left, input_type[0].right, output_type.right
            return Diagram.fc(l, m, r)
        elif self == self.BACKWARD_COMPOSITION:
            assert input_type[0].right == input_type[1].left
            l, m, r = output_type.left, input_type[0].right, output_type.right
            return Diagram.bc(l, m, r)
        elif self == self.FORWARD_CROSSED_COMPOSITION:
            assert input_type[0].right == input_type[1].right
            l, m, r = output_type.right, input_type[0].right, output_type.left
            return Diagram.fx(l, m, r)
        elif self == self.BACKWARD_CROSSED_COMPOSITION:
            assert input_type[0].left == input_type[1].left
            l, m, r = output_type.right, input_type[0].left, output_type.left
            return Diagram.bx(l, m, r)
        elif self == self.REMOVE_PUNCTUATION_LEFT:
            return RPL(input_type[:1], output_type)
        elif self == self.REMOVE_PUNCTUATION_RIGHT:
            return RPR(output_type, input_type[1:])
        elif self == self.FORWARD_TYPE_RAISING:
            return Diagram.curry(Diagram.ba(output_type.right.left,
                                            output_type.left))
        elif self == self.BACKWARD_TYPE_RAISING:
            return Diagram.curry(Diagram.fa(output_type.right,
                                            output_type.left.right), left=True)
        elif self == self.CONJUNCTION:
            left, right = input_type[:1], input_type[1:]
            if CCGAtomicType.conjoinable(left):
                return Diagram.fa(output_type, right)
            elif CCGAtomicType.conjoinable(right):
                return Diagram.ba(left, output_type)
            else:
                raise CCGRuleUseError(self, 'No conjunction found.')
        elif self == self.LEXICAL:
            return Id(output_type)
        raise CCGRuleUseError(self, 'Unknown CCG rule.')
