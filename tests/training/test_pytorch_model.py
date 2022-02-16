from lambeq.training import PytorchModel

from discopy import Cup, Dim, Word
from discopy.quantum.circuit import Id

from torch import Size
from torch.nn import Parameter

from lambeq import AtomicType, SpiderAnsatz

N = AtomicType.NOUN
S = AtomicType.SENTENCE


def test_init():
    ansatz = SpiderAnsatz({N: Dim(2), S: Dim(2)})
    diagrams = [
        ansatz((Word("Alice", N) @ Word("runs", N >> S) >> Cup(N, N.r) @ Id(S)))
    ]

    model = PytorchModel(diagrams)
    assert len(model.weights) == 2
    assert all(isinstance(x, Parameter) for x in model.weights)

def test_forward():
    s_dim = 2
    ansatz = SpiderAnsatz({N: Dim(2), S: Dim(s_dim)})
    diagrams = [
        ansatz((Word("Alice", N) @ Word("runs", N >> S) >> Cup(N, N.r) @ Id(S)))
    ]
    instance = PytorchModel(diagrams)
    pred = instance.forward(diagrams)
    assert pred.size() == Size([len(diagrams), s_dim])
