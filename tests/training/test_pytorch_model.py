import os
import pickle
import pytest
from copy import deepcopy

from lambeq.training import PytorchModel

from discopy import Box, Cap, Cup, Dim, Swap, Word
from discopy.tensor import Id as tensor_Id
from discopy.quantum.circuit import Id
from discopy.rigid import Spider

import numpy as np
from torch import Size
from torch.nn import Parameter

from lambeq import AtomicType, SpiderAnsatz, Symbol
from lambeq.training import PytorchModel

N = AtomicType.NOUN
S = AtomicType.SENTENCE


def test_init():
    ansatz = SpiderAnsatz({N: Dim(2), S: Dim(2)})
    diagrams = [
        ansatz((Word("Alice", N) @ Word("runs", N >> S) >> Cup(N, N.r) @ Id(S)))
    ]

    model = PytorchModel.initialise_symbols(diagrams)
    model.initialise_weights()
    assert len(model.weights) == 2
    assert all(isinstance(x, Parameter) for x in model.weights)

def test_forward():
    s_dim = 2
    ansatz = SpiderAnsatz({N: Dim(2), S: Dim(s_dim)})
    diagrams = [
        ansatz((Word("Alice", N) @ Word("runs", N >> S) >> Cup(N, N.r) @ Id(S)))
    ]
    instance = PytorchModel.initialise_symbols(diagrams)
    instance.initialise_weights()
    pred = instance.forward(diagrams)
    assert pred.size() == Size([len(diagrams), s_dim])


def test_pickling():
    phi = Symbol('phi', size=123)
    diagram = (
        Box("box1", Dim(2), Dim(2), data=phi)
        >> Spider(1, 2, Dim(2))
        >> Swap(Dim(2), Dim(2))
        >> tensor_Id(Dim(2))
        @ (tensor_Id(Dim(2)) @ Cap(Dim(2), Dim(2)) >> Cup(Dim(2), Dim(2)) @ tensor_Id(Dim(2)))
    )
    deepcopied_diagram = deepcopy(diagram)
    pickled_diagram = pickle.loads(pickle.dumps(diagram))
    assert pickled_diagram == diagram
    pickled_diagram._data = 'new data'
    for box in pickled_diagram.boxes:
        box._name = 'Bob'
        box._data = ['random', 'data']
    assert diagram == deepcopied_diagram
    assert diagram != pickled_diagram
    assert deepcopied_diagram != pickled_diagram


def test_initialise_weights_error():
    with pytest.raises(ValueError):
        model = PytorchModel()
        model.initialise_weights()

def test_get_diagram_output_error():
    N = AtomicType.NOUN
    S = AtomicType.SENTENCE
    ansatz = SpiderAnsatz({N: Dim(2), S: Dim(2)})
    diagram = ansatz((Word("Alice", N) @ Word("runs", N >> S) >> Cup(N, N.r) @ Id(S)))
    with pytest.raises(KeyError):
        model = PytorchModel()
        model.get_diagram_output([diagram])

def test_checkpoint_loading():
    N = AtomicType.NOUN
    S = AtomicType.SENTENCE
    ansatz = SpiderAnsatz({N: Dim(2), S: Dim(2)})
    diagram = ansatz((Word("Alice", N) @ Word("runs", N >> S) >> Cup(N, N.r) @ Id(S)))
    model = PytorchModel.initialise_symbols([diagram])
    model.initialise_weights()

    checkpoint = {'model_weights': model.weights,
                  'model_symbols': model.symbols,
                  'model_state_dict': model.state_dict()}
    with open('model.lt', 'wb') as f:
        pickle.dump(checkpoint, f)

    model_new = PytorchModel.load_from_checkpoint('model.lt')
    os.remove('model.lt')
    assert len(model_new.weights)==len(model.weights)
    assert model_new.symbols==model.symbols
    assert np.all(model([diagram]).detach().numpy()==model_new([diagram]).detach().numpy())

def test_checkpoint_loading_errors():
    checkpoint = {'model_weights': np.array([1,2,3])}
    with open('model.lt', 'wb') as f:
        pickle.dump(checkpoint, f)
    with pytest.raises(KeyError):
        _ = PytorchModel.load_from_checkpoint('model.lt')
    os.remove('model.lt')

def test_checkpoint_loading_file_not_found_errors():
    try:
        os.remove('model.lt')
    except:
        pass
    with pytest.raises(FileNotFoundError):
        _ = PytorchModel.load_from_checkpoint('model.lt')
