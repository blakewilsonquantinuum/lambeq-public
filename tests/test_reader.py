import pytest

from discopy import Word
from discopy.rigid import Diagram, Id

from discoket.core.types import AtomicType
from discoket.reader import (Reader, box_stairs_reader,
                             box_stairs_with_discard_reader, cups_reader,
                             spiders_reader)


@pytest.fixture
def sentence():
    return 'This is a sentence'


@pytest.fixture
def words(sentence):
    words = sentence.split()
    assert len(words) == 4
    return words


def test_reader():
    with pytest.raises(TypeError):
        Reader()


def test_box_stair_reader(sentence, words):
    S = AtomicType.SENTENCE
    combining_diagram = box_stairs_reader.combining_diagram
    assert combining_diagram.dom == S @ S and combining_diagram.cod == S

    expected_diagram = (Diagram.tensor(*(Word(word, S) for word in words)) >>
                        combining_diagram @ Id(S @ S) >>
                        combining_diagram @ Id(S) >>
                        combining_diagram)
    assert (box_stairs_reader.sentences2diagrams([sentence])[0] ==
            box_stairs_reader.sentence2diagram(sentence) == expected_diagram)


def test_other_readers(sentence):
    # since all the readers share behaviour, just test that they don't fail
    assert box_stairs_with_discard_reader.sentence2diagram(sentence)
    assert cups_reader.sentence2diagram(sentence)
    assert spiders_reader.sentence2diagram(sentence)
