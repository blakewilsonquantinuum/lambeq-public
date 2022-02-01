from unittest.mock import patch

from discopy import Cup, Dim, Word
from discopy.quantum.circuit import Id
from lambeq.ansatz import Symbol
from lambeq.core.types import AtomicType
from lambeq.tensor import SpiderAnsatz
from lambeq.training import Model

N = AtomicType.NOUN
S = AtomicType.SENTENCE


@patch.multiple(Model, __abstractmethods__=set())
def test_instance():
    instance = Model()


@patch.multiple(Model, __abstractmethods__=set())
def test_init():
    instance = Model()
    assert instance.vocab == [] and instance.word_params == []


@patch.multiple(Model, __abstractmethods__=set())
def test_prepare_vocab():
    instance = Model()
    ansatz = SpiderAnsatz({N: Dim(2), S: Dim(2)})
    diagrams = [
        ansatz((Word("Alice", N) @ Word("runs", N >> S) >> Cup(N, N.r) @ Id(S)))
    ]
    instance.prepare_vocab(diagrams)
    assert len(instance.vocab) == 2
    assert all(isinstance(x, Symbol) for x in instance.vocab)
