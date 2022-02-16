import numpy as np
from discopy import Cup, Word
from discopy.quantum.circuit import Id
from pytket.extensions.qiskit import AerBackend

from lambeq import AtomicType, IQPAnsatz, QuantumModel

N = AtomicType.NOUN
S = AtomicType.SENTENCE

backend = AerBackend()

backend_config = {
    'backend': backend,
    'compilation': backend.default_compilation_pass(2),
    'shots': 8192  # maximum recommended shots, reduces sampling error
}

ansatz = IQPAnsatz({N: 1, S: 1}, n_layers=1)
diagrams = [
    ansatz((Word("Alice", N) @ Word("runs", N >> S) >> Cup(N, N.r) @ Id(S)))
]

def test_init():
    model = QuantumModel(diagrams, backend_config, seed=0)
    assert len(model.weights) == 4
    assert isinstance(model.weights, np.ndarray)

def test_forward():
    model = QuantumModel(diagrams, backend_config, seed=0)
    pred = model.forward(diagrams)
    assert pred.shape == (len(diagrams), 2)
