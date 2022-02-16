from lambeq import Dataset

data = [1, 2, 3, 4]
targets = [5, 6, 7, 8]


def test_get_item():
    dataset = Dataset(data, targets, batch_size=2, shuffle=False, seed=0)
    index = 0
    x, y = dataset[index]

    assert x == data[index]
    assert y == targets[index]


def test_len():
    dataset = Dataset(data, targets, batch_size=2, shuffle=False, seed=0)
    assert len(dataset) == len(data)
    assert len(dataset) == len(targets)


def test_batch_gen():
    dataset = Dataset(data, targets, batch_size=2, shuffle=True, seed=0)

    new_data = []
    new_targets = []
    for batch in dataset:
        new_data.append(batch[0])
        new_targets.append(batch[1])

    assert new_data == [[3, 1], [2, 4]]
    assert new_targets == [[7, 5], [6, 8]]


def test_full_batch():
    dataset = Dataset(data, targets, batch_size=2, shuffle=False, seed=0)
    x, y = dataset[:]
    assert x == data
    assert y == targets


def test_shuffle():
    data = list(range(100))
    targets = list(range(100))
    new_data, new_targets = Dataset.shuffle_data(data, targets)
    assert new_data == new_targets
