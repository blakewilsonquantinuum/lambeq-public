import numpy as np

from discopy import Cup, Word
from discopy.quantum.circuit import Id

from lambeq import AtomicType, IQPAnsatz, SPSAOptimiser

N = AtomicType.NOUN
S = AtomicType.SENTENCE

ansatz = IQPAnsatz({N: 1, S: 1}, n_layers=1, n_single_qubit_params=1)

diagrams = [
    ansatz((Word("Alice", N) @ Word("runs", N >> S) >> Cup(N, N.r) @ Id(S))),
    ansatz((Word("Alice", N) @ Word("walks", N >> S) >> Cup(N, N.r) @ Id(S)))
]

from lambeq.training.model import Model


class ModelDummy(Model):
    def __init__(self, diagrams, seed = None) -> None:
        super().__init__(diagrams, seed)
        self.weights = np.array([1.,2.,3.])
    def get_diagram_output(self):
        pass
    def forward(self, x):
        return self.weights.sum()

loss = lambda yhat, y: np.abs(yhat-y).sum()**2

def test_init():
    model = ModelDummy(diagrams)
    optim = SPSAOptimiser(model,
                          hyperparams={'a': 0.01, 'c': 0.1, 'A':0.001},
                          loss_fn= loss,
                          bounds=[[0, 10]]*len(model.weights))
    assert optim.alpha
    assert optim.gamma
    assert optim.current_sweep
    assert optim.A
    assert optim.a
    assert optim.c
    assert optim.ak
    assert optim.ck
    assert optim.project
    assert optim.rng

def test_backward():
    model = ModelDummy(diagrams)
    optim = SPSAOptimiser(model,
                          hyperparams={'a': 0.01, 'c': 0.1, 'A':0.001},
                          loss_fn= loss,
                          bounds=[[0, 10]]*len(model.weights),
                          seed=0)
    optim.backward(([diagrams[0]], [0]))
    assert np.array_equal(optim.gradient.round(5), np.array([12, 12, 0]))
    assert np.array_equal(model.weights, np.array([1.,2.,3.]))

def test_step():
    model = ModelDummy(diagrams)
    optim = SPSAOptimiser(model,
                          hyperparams={'a': 0.01, 'c': 0.1, 'A':0.001},
                          loss_fn= loss,
                          bounds=[[0, 10]]*len(model.weights),
                          seed=0)
    step_counter = optim.current_sweep
    optim.backward(([diagrams[0]], [0]))
    optim.step()
    assert np.array_equal(model.weights.round(4), np.array([0.8801,1.8801,3.]))
    assert optim.current_sweep == step_counter+1
    assert round(optim.ak,5) == 0.00659
    assert round(optim.ck,5) == 0.09324

def test_project():
    model = ModelDummy(diagrams)
    model.weights = np.array([0, 10, 0])
    optim = SPSAOptimiser(model,
                          hyperparams={'a': 0.01, 'c': 0.1, 'A':0.001},
                          loss_fn= loss,
                          bounds=[[0, 10]]*len(model.weights),
                          seed=0)
    optim.backward((diagrams, np.array([0, 0])))
    assert np.array_equal(
        optim.gradient.round(1), np.array([241.2, 241.2, 241.2]))
