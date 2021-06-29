import pytest

from discopy import Word
from discopy.rigid import Diagram, Id, Ty

from discoket.split import MPSSplitter, SpiderSplitter


@pytest.fixture
def diagram():
    cod = Ty(*'abcde')
    return (Word('big', cod.l) @ Word('words', cod @ cod) @
            Diagram.cups(cod.l, cod) @ Id(cod))


def test_mps_splitter(diagram):
    Z = Ty('Z')

    with pytest.raises(ValueError):
        MPSSplitter(max_order=2, bond_type=Z)

    for i in range(3, 6):
        splitter = MPSSplitter(max_order=i, bond_type=Z)
        split_diagram = splitter(diagram)
        for box in split_diagram.boxes:
            assert len(box.cod) <= i


def test_spider_splitter(diagram):
    with pytest.raises(ValueError):
        MPSSplitter(max_order=1)

    for i in range(2, 6):
        splitter = SpiderSplitter(max_order=i)
        split_diagram = splitter(diagram)
        for box in split_diagram.boxes:
            assert len(box.cod) <= i
