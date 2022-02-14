import numpy as np
from discopy import Cup, Word
from discopy.quantum.circuit import Id
from lambeq.core.types import AtomicType
from lambeq.circuit import IQPAnsatz
from lambeq.training import Dataset, ECSQuantumModel, QuantumTrainer, SPSAOptimiser

N = AtomicType.NOUN
S = AtomicType.SENTENCE
EPOCHS = 1

train_diagrams = [
    (Word("Alice", N) @ Word("runs", N >> S) >> Cup(N, N.r) @ Id(S)),
    (Word("Alice", N) @ Word("waits", N >> S) >> Cup(N, N.r) @ Id(S)),
    (Word("Bob", N) @ Word("runs", N >> S) >> Cup(N, N.r) @ Id(S)),
    (Word("Bob", N) @ Word("eats", N >> S) >> Cup(N, N.r) @ Id(S)),
]
train_targets = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])

dev_diagrams = [
    (Word("Alice", N) @ Word("eats", N >> S) >> Cup(N, N.r) @ Id(S)),
    (Word("Bob", N) @ Word("waits", N >> S) >> Cup(N, N.r) @ Id(S)),
]
dev_targets = np.array([[0, 1], [1, 0]])

loss = lambda y_hat, y: -np.sum(y * np.log(y_hat)) / len(y)
acc = lambda y_hat, y: np.sum(np.round(y_hat) == y) / len(y) / 2

def test_trainer():
    ansatz = IQPAnsatz({N: 1, S: 1}, n_layers=1)
    train_circuits = [ansatz(d) for d in train_diagrams]
    dev_circuits = [ansatz(d) for d in dev_diagrams]

    model = ECSQuantumModel(train_circuits + dev_circuits, seed=42)

    trainer = QuantumTrainer(
        model=model,
        loss_function=loss,
        optimizer=SPSAOptimiser,
        optim_hyperparams={'a': 0.02, 'c': 0.06, 'A':0.01*EPOCHS},
        epochs=EPOCHS,
        evaluate_functions={"acc": acc},
        use_tensorboard=True,
        verbose=False,
        seed=42,
    )

    train_dataset = Dataset(
            train_circuits,
            list(train_targets), # dataset requires list of tensors
            seed=0
        )

    val_dataset = Dataset(
            dev_circuits,
            list(dev_targets),
            seed=0
        )

    trainer.fit(train_dataset, val_dataset) # dataset requires list of tensors

    assert len(trainer.train_costs) == EPOCHS
    assert len(trainer.val_results["acc"]) == EPOCHS
