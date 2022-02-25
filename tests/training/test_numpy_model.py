import numpy as np
from discopy import Cup, Word
from discopy.quantum.circuit import Id

from lambeq import AtomicType, IQPAnsatz, NumpyModel

N = AtomicType.NOUN
S = AtomicType.SENTENCE

s_dim = 2
ansatz = IQPAnsatz({N: 1, S: 1}, n_layers=1)
diagrams = [
    ansatz((Word("Alice", N) @ Word("runs", N >> S) >> Cup(N, N.r) @ Id(S)))
]

def test_init():
    model = NumpyModel(diagrams, seed=0)
    assert len(model.weights) == 4
    assert isinstance(model.weights, np.ndarray)

def test_forward():
    instance = NumpyModel(diagrams, seed=0)
    pred = instance.forward(diagrams)
    assert pred.shape == (len(diagrams), s_dim)
