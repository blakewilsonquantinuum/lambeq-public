from discopy import Word
from discopy.rigid import Cup
import tensornetwork as tn
import torch

from discoket.core.types import AtomicType, Spider
from discoket.tensor import TensorDiagram


def tensor_for_box(i, box, diag):
    return torch.ones([4]*len(box.cod))


def test_tensor():
    tn.set_default_backend('pytorch')

    N = AtomicType.NOUN
    S = AtomicType.SENTENCE

    diagram = ((Word('This', N) @ Word('is_1', N >> S) @ Word('is_2', S << N) @
                Word('a', N << N) @ Word('sentence', N)) >>
               Cup(N, N.r) @ Spider(2, 1, S) @ Cup(N.l, N) @ Cup(N.l, N))

    td = TensorDiagram(diagram)
    assert torch.all(td(closure=tensor_for_box) == torch.tensor([64] * 4))
