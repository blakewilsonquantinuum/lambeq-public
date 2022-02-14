import numpy as np
from discopy import Cup, Word
from discopy.quantum.circuit import Id

from lambeq.circuit import IQPAnsatz
from lambeq.core.types import AtomicType
from lambeq.training import ECSQuantumModel

N = AtomicType.NOUN
S = AtomicType.SENTENCE

s_dim = 2
ansatz = IQPAnsatz({N: 1, S: 1}, n_layers=1)
diagrams = [
    ansatz((Word("Alice", N) @ Word("runs", N >> S) >> Cup(N, N.r) @ Id(S)))
]

def test_init():
    model = ECSQuantumModel(diagrams, seed=0)
    assert len(model.weights) == 4
    assert isinstance(model.weights, np.ndarray)

def test_forward():
    instance = ECSQuantumModel(diagrams, seed=0)
    pred = instance.forward(diagrams)
    assert pred.shape == (len(diagrams), s_dim)
