import os
import pickle
import pytest

import numpy as np
from jax import numpy as jnp
from discopy import Cup, Word, Tensor
from discopy.quantum.circuit import Id

from lambeq import AtomicType, IQPAnsatz, NumpyModel

def test_init():
    Tensor.np = np

    N = AtomicType.NOUN
    S = AtomicType.SENTENCE

    ansatz = IQPAnsatz({N: 1, S: 1}, n_layers=1)
    diagrams = [ansatz((Word("Alice", N) @ Word("runs", N >> S) >> Cup(N, N.r) @ Id(S)))]
    model = NumpyModel.initialise_symbols(diagrams)
    model.initialise_weights()
    assert len(model.weights) == 4
    assert isinstance(model.weights, np.ndarray)

def test_forward():
    Tensor.np = np

    N = AtomicType.NOUN
    S = AtomicType.SENTENCE

    s_dim = 2
    ansatz = IQPAnsatz({N: 1, S: 1}, n_layers=1)
    diagrams = [ansatz((Word("Alice", N) @ Word("runs", N >> S) >> Cup(N, N.r) @ Id(S)))]
    model = NumpyModel.initialise_symbols(diagrams)
    model.initialise_weights()
    pred = model.forward(diagrams)
    assert pred.shape == (len(diagrams), s_dim)

def test_make_lambda_error():
    Tensor.np = np
    N = AtomicType.NOUN
    S = AtomicType.SENTENCE
    ansatz = IQPAnsatz({N: 1, S: 1}, n_layers=1)
    diagram = ansatz((Word("Alice", N) @ Word("runs", N >> S) >> Cup(N, N.r) @ Id(S)))
    with pytest.raises(ValueError):
        model = NumpyModel()
        model._make_lambda(diagram)

def test_initialise_weights_error():
    Tensor.np = np
    with pytest.raises(ValueError):
        model = NumpyModel()
        model.initialise_weights()

def test_get_diagram_output_error():
    Tensor.np = np
    N = AtomicType.NOUN
    S = AtomicType.SENTENCE
    ansatz = IQPAnsatz({N: 1, S: 1}, n_layers=1)
    diagram = ansatz((Word("Alice", N) @ Word("runs", N >> S) >> Cup(N, N.r) @ Id(S)))
    with pytest.raises(ValueError):
        model = NumpyModel()
        model.get_diagram_output([diagram])

def test_jax_usage():
    Tensor.np = jnp
    N = AtomicType.NOUN
    S = AtomicType.SENTENCE
    ansatz = IQPAnsatz({N: 1, S: 1}, n_layers=1)
    diagram = ansatz((Word("Alice", N) @ Word("runs", N >> S) >> Cup(N, N.r) @ Id(S)))
    model = NumpyModel.initialise_symbols([diagram])
    diag_f = model._make_lambda(diagram)
    assert type(diag_f).__name__ == 'CompiledFunction'  # TODO needs better solution
    Tensor.np = np

def test_checkpoint_loading():
    N = AtomicType.NOUN
    S = AtomicType.SENTENCE
    ansatz = IQPAnsatz({N: 1, S: 1}, n_layers=1)
    diagram = ansatz((Word("Alice", N) @ Word("runs", N >> S) >> Cup(N, N.r) @ Id(S)))
    model = NumpyModel.initialise_symbols([diagram])
    model.initialise_weights()

    checkpoint = {'model_weights': model.weights,
                  'model_symbols': model.symbols}
    with open('model.lt', 'wb') as f:
        pickle.dump(checkpoint, f)

    model_new = NumpyModel.load_from_checkpoint('model.lt')
    os.remove('model.lt')
    assert np.all(model.weights==model_new.weights)
    assert model_new.symbols==model.symbols
    assert np.all(model([diagram])==model_new([diagram]))


def test_checkpoint_loading_errors():
    checkpoint = {'model_weights': np.array([1,2,3])}
    with open('model.lt', 'wb') as f:
        pickle.dump(checkpoint, f)
    with pytest.raises(KeyError):
        _ = NumpyModel.load_from_checkpoint('model.lt')
    os.remove('model.lt')

def test_checkpoint_loading_file_not_found_errors():
    try:
        os.remove('model.lt')
    except:
        pass
    with pytest.raises(FileNotFoundError):
        _ = NumpyModel.load_from_checkpoint('model.lt')
