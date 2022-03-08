import os
import pickle
import pytest

import numpy as np
from discopy import Cup, Word, Tensor
from discopy.quantum.circuit import Id
from pytket.extensions.qiskit import AerBackend

from lambeq import AtomicType, IQPAnsatz, TketModel

Tensor.np = np

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
    model = TketModel.initialise_symbols(diagrams, backend_config=backend_config)
    model.initialise_weights()
    assert len(model.weights) == 4
    assert isinstance(model.weights, np.ndarray)

def test_forward():
    model = TketModel.initialise_symbols(diagrams, backend_config=backend_config)
    model.initialise_weights()
    pred = model.forward(diagrams)
    assert pred.shape == (len(diagrams), 2)
    pred2 = model.forward(2*diagrams)
    assert pred2.shape == (2*len(diagrams), 2)

def test_initialise_weights_error():
    Tensor.np = np
    with pytest.raises(ValueError):
        model = TketModel(backend_config=backend_config)
        model.initialise_weights()

def test_get_diagram_output_error():
    Tensor.np = np
    N = AtomicType.NOUN
    S = AtomicType.SENTENCE
    ansatz = IQPAnsatz({N: 1, S: 1}, n_layers=1)
    diagram = ansatz((Word("Alice", N) @ Word("runs", N >> S) >> Cup(N, N.r) @ Id(S)))
    with pytest.raises(ValueError):
        model = TketModel(backend_config=backend_config)
        model.get_diagram_output([diagram])

def test_checkpoint_loading():
    checkpoint = {'model_weights': np.array([1,2,3]),
                  'model_symbols': ['a', 'b', 'c']}
    with open('model.lt', 'wb') as f:
        pickle.dump(checkpoint, f)
    model = TketModel.load_from_checkpoint('model.lt',
                                           backend_config=backend_config)
    os.remove('model.lt')
    assert np.all(model.weights==checkpoint['model_weights'])
    assert model.symbols==checkpoint['model_symbols']


def test_checkpoint_loading_errors():
    checkpoint = {'model_weights': np.array([1,2,3])}
    with open('model.lt', 'wb') as f:
        pickle.dump(checkpoint, f)
    with pytest.raises(KeyError):
        _ = TketModel.load_from_checkpoint('model.lt',
                                           backend_config=backend_config)
    os.remove('model.lt')

def test_checkpoint_loading_file_not_found_errors():
    try:
        os.remove('model.lt')
    except:
        pass
    with pytest.raises(FileNotFoundError):
        _ = TketModel.load_from_checkpoint('model.lt',
                                           backend_config=backend_config)

def test_missing_field_error():
    with pytest.raises(KeyError):
        _ = TketModel(backend_config={})

def test_missing_backend_error():
    with pytest.raises(KeyError):
        _ = TketModel()