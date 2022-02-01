import pytest

from lambeq.training import PytorchModel

from discopy import Cup, Dim, Word
from discopy.quantum.circuit import Id
from lambeq.core.types import AtomicType
from lambeq.tensor import SpiderAnsatz
from torch.nn import Parameter
from torch import Size

N = AtomicType.NOUN
S = AtomicType.SENTENCE



def test_tensorise():
    instance = PytorchModel()
    ansatz = SpiderAnsatz({N: Dim(2), S: Dim(2)})
    diagrams = [
        ansatz((Word("Alice", N) @ Word("runs", N >> S) >> Cup(N, N.r) @ Id(S)))
    ]
    instance.prepare_vocab(diagrams)
    instance.tensorise()
    assert len(instance.word_params) == 2
    assert all(isinstance(x, Parameter) for x in instance.word_params)

def test_tensorise_error():
    instance = PytorchModel()
    with pytest.raises(ValueError):
        instance.tensorise()

def test_lambdify_error():
    instance = PytorchModel()
    with pytest.raises(ValueError):
        instance.lambdify([])

def test_forward():
    instance = PytorchModel()
    s_dim = 2
    ansatz = SpiderAnsatz({N: Dim(2), S: Dim(s_dim)})
    diagrams = [
        ansatz((Word("Alice", N) @ Word("runs", N >> S) >> Cup(N, N.r) @ Id(S)))
    ]
    instance.prepare_vocab(diagrams)
    instance.tensorise()
    pred = instance.forward(diagrams)
    assert pred.size() == Size([len(diagrams), s_dim])
