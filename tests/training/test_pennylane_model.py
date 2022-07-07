import pickle
import pytest
from copy import deepcopy
from unittest.mock import mock_open, patch

import numpy as np
import torch
from torch import Size
from torch.nn import Parameter

from discopy import Cup,  Word
from discopy.quantum.circuit import Id
from lambeq import AtomicType, Dataset, IQPAnsatz, PennyLaneModel, PytorchTrainer

N = AtomicType.NOUN
S = AtomicType.SENTENCE


def test_init():
    ansatz = IQPAnsatz({AtomicType.NOUN: 1, AtomicType.SENTENCE: 1},
                       n_layers=1, n_single_qubit_params=1)
    diagrams = [
        ansatz((Word("Alice", N) @ Word("runs", N >> S) >> Cup(N, N.r) @ Id(S)))
    ]

    model = PennyLaneModel.from_diagrams(diagrams)
    model.initialise_weights()
    assert len(model.weights) == 2
    assert all(isinstance(x, Parameter) for x in model.weights)


def test_forward():
    s_dim = 2
    ansatz = IQPAnsatz({AtomicType.NOUN: 1, AtomicType.SENTENCE: 1},
                       n_layers=1, n_single_qubit_params=3)
    diagrams = [
        ansatz((Word("Alice", N) @ Word("runs", N >> S) >> Cup(N, N.r) @ Id(S)))
    ]
    print(type(diagrams[0]))
    instance = PennyLaneModel.from_diagrams(diagrams)
    instance.initialise_weights()
    pred = instance.forward(diagrams)
    assert pred.size() == Size([len(diagrams), s_dim])


def test_initialise_weights_error():
    with pytest.raises(ValueError):
        model = PennyLaneModel()
        model.initialise_weights()


def test_get_diagram_output_error():
    N = AtomicType.NOUN
    S = AtomicType.SENTENCE
    ansatz = IQPAnsatz({AtomicType.NOUN: 1, AtomicType.SENTENCE: 1},
                       n_layers=1, n_single_qubit_params=3)
    diagram = ansatz((Word("Alice", N) @ Word("runs", N >> S) >> Cup(N, N.r) @ Id(S)))
    with pytest.raises(KeyError):
        model = PennyLaneModel()
        model.get_diagram_output([diagram])


def test_checkpoint_loading():
    N = AtomicType.NOUN
    S = AtomicType.SENTENCE
    ansatz = IQPAnsatz({AtomicType.NOUN: 1, AtomicType.SENTENCE: 1},
                       n_layers=1, n_single_qubit_params=3)
    diagram = ansatz((Word("Alice", N) @ Word("runs", N >> S) >> Cup(N, N.r) @ Id(S)))
    model = PennyLaneModel.from_diagrams([diagram])
    model.initialise_weights()

    checkpoint = {'model_weights': model.weights,
                  'model_symbols': model.symbols,
                  'model_circuits': model.circuit_map,
                  'model_state_dict': model.state_dict()}
    with patch('lambeq.training.checkpoint.open', mock_open(read_data=pickle.dumps(checkpoint))) as m, \
            patch('lambeq.training.checkpoint.os.path.exists', lambda x: True) as p:
        model_new = PennyLaneModel.from_checkpoint('model.lt')
        assert len(model_new.weights) == len(model.weights)
        assert model_new.symbols == model.symbols
        assert np.all(model([diagram]).detach().numpy() == model_new([diagram]).detach().numpy())
        m.assert_called_with('model.lt', 'rb')


def test_checkpoint_loading_errors():
    checkpoint = {'model_weights': np.array([1,2,3])}
    with patch('lambeq.training.checkpoint.open', mock_open(read_data=pickle.dumps(checkpoint))) as m, \
            patch('lambeq.training.checkpoint.os.path.exists', lambda x: True) as p:
        with pytest.raises(KeyError):
            _ = PennyLaneModel.from_checkpoint('model.lt')
        m.assert_called_with('model.lt', 'rb')


def test_checkpoint_loading_file_not_found_errors():
    with patch('lambeq.training.checkpoint.open', mock_open(read_data='Not a valid checkpoint.')) as m, \
            patch('lambeq.training.checkpoint.os.path.exists', lambda x: False) as p:
        with pytest.raises(FileNotFoundError):
            _ = PennyLaneModel.from_checkpoint('model.lt')
        m.assert_not_called()


def test_with_pytorch_trainer(tmp_path):
    EPOCHS = 1
    sig = torch.sigmoid
    acc = lambda y_hat, y: torch.sum(torch.eq(torch.round(sig(y_hat)), y))/len(y)/2

    train_diagrams = [
        (Word("Alice", N) @ Word("runs", N >> S) >> Cup(N, N.r) @ Id(S)),
        (Word("Alice", N) @ Word("waits", N >> S) >> Cup(N, N.r) @ Id(S)),
        (Word("Bob", N) @ Word("runs", N >> S) >> Cup(N, N.r) @ Id(S)),
        (Word("Bob", N) @ Word("eats", N >> S) >> Cup(N, N.r) @ Id(S)),
    ]
    train_targets = [[1, 0], [0, 1], [0, 1], [1, 0]]

    dev_diagrams = [
        (Word("Alice", N) @ Word("eats", N >> S) >> Cup(N, N.r) @ Id(S)),
        (Word("Bob", N) @ Word("waits", N >> S) >> Cup(N, N.r) @ Id(S)),
    ]
    dev_targets = [[0, 1], [1, 0]]

    ansatz = IQPAnsatz({N: 1, S: 1}, n_layers=1, n_single_qubit_params=3)
    train_circuits = [ansatz(d) for d in train_diagrams]
    dev_circuits = [ansatz(d) for d in dev_diagrams]

    model = PennyLaneModel.from_diagrams(train_circuits + dev_circuits)

    log_dir = tmp_path / 'test_runs'
    log_dir.mkdir()

    trainer = PytorchTrainer(
        model=model,
        loss_function=torch.nn.BCEWithLogitsLoss(),
        optimizer=torch.optim.AdamW,
        learning_rate=3e-3,
        epochs=EPOCHS,
        evaluate_functions={"acc": acc},
        evaluate_on_train=True,
        use_tensorboard=True,
        log_dir=log_dir,
        verbose='suppress',
        seed=0
    )

    train_dataset = Dataset(train_circuits, train_targets)
    val_dataset = Dataset(dev_circuits, dev_targets)

    trainer.fit(train_dataset, val_dataset)

    assert len(trainer.train_costs) == EPOCHS
    assert len(trainer.val_results["acc"]) == EPOCHS
