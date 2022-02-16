from discopy import Cup, Dim, Word
from discopy.quantum.circuit import Id
import torch

from lambeq import AtomicType, SpiderAnsatz, Dataset, PytorchTrainer, PytorchModel

N = AtomicType.NOUN
S = AtomicType.SENTENCE
EPOCHS = 1


def accuracy(preds, targets):
    hits = 0
    for i in range(len(preds)):
        target = targets[i]
        pred = preds[i]
        if torch.argmax(target) == torch.argmax(pred):
            hits += 1
    return hits / len(preds)


train_diagrams = [
    (Word("Alice", N) @ Word("runs", N >> S) >> Cup(N, N.r) @ Id(S)),
    (Word("Alice", N) @ Word("waits", N >> S) >> Cup(N, N.r) @ Id(S)),
    (Word("Bob", N) @ Word("runs", N >> S) >> Cup(N, N.r) @ Id(S)),
    (Word("Bob", N) @ Word("eats", N >> S) >> Cup(N, N.r) @ Id(S)),
]
train_targets = torch.as_tensor([[1, 0], [0, 1], [0, 1], [1, 0]], dtype=torch.double)

dev_diagrams = [
    (Word("Alice", N) @ Word("eats", N >> S) >> Cup(N, N.r) @ Id(S)),
    (Word("Bob", N) @ Word("waits", N >> S) >> Cup(N, N.r) @ Id(S)),
]
dev_targets = torch.as_tensor([[0, 1], [1, 0]], dtype=torch.double)


def test_trainer():
    ansatz = SpiderAnsatz({N: Dim(2), S: Dim(2)})
    train_circuits = [ansatz(d) for d in train_diagrams]
    dev_circuits = [ansatz(d) for d in dev_diagrams]

    model = PytorchModel(train_circuits + dev_circuits, seed=42)

    trainer = PytorchTrainer(
        model=model,
        loss_function=torch.nn.BCEWithLogitsLoss(),
        optimizer=torch.optim.AdamW,
        learning_rate=3e-3,
        epochs=EPOCHS,
        evaluate_functions={"acc": accuracy},
        use_tensorboard=True,
    )

    train_dataset = Dataset(
            train_circuits,
            list(torch.unbind(train_targets)),  # dataset requires list of tensors
            seed=0
        )

    val_dataset = Dataset(
            dev_circuits,
            list(torch.unbind(dev_targets)),
            seed=0
        )

    trainer.fit(train_dataset, val_dataset)  # dataset requires list of tensors

    assert len(trainer.train_costs) == EPOCHS
    assert len(trainer.val_results["acc"]) == EPOCHS
