import pytest
from requests.exceptions import MissingSchema

from lambeq.backend.grammar import Box, Id, Spider, Word
from lambeq.backend.quantum import Bra, CX, Id as QId, Ket, qubit

from lambeq import (AtomicType, BobcatParser, IQPAnsatz, DependencyReader,
                    TreeReader, TreeReaderMode, VerbosityLevel, WebParser,
                    cups_reader, spiders_reader, stairs_reader)


@pytest.fixture
def sentence():
    return 'This is a sentence'


@pytest.fixture
def words(sentence):
    words = sentence.split()
    assert len(words) == 4
    return words


@pytest.fixture
def parser():
    return BobcatParser(verbose=VerbosityLevel.SUPPRESS.value)


def test_spiders_reader(sentence, words):
    S = AtomicType.SENTENCE

    expected_diagram = (Id().tensor(*(Word(word, S) for word in words)) >>
                        Spider(S, len(words), 1))
    assert (spiders_reader.sentences2diagrams([sentence])[0] ==
            spiders_reader.sentence2diagram(sentence) == expected_diagram)


def test_spiders_reader_tokenised(sentence, words):
    S = AtomicType.SENTENCE

    expected_diagram = (Id().tensor(*(Word(word, S) for word in words)) >>
                        Spider(S, len(words), 1))
    assert (spiders_reader.sentences2diagrams([sentence.split()], tokenised=True)[0] ==
            spiders_reader.sentence2diagram(sentence.split(), tokenised=True) ==
            expected_diagram)


def test_spiders_reader_circuit(sentence, words):
    S = AtomicType.SENTENCE

    ansatz = IQPAnsatz({S: 1}, n_layers=1, n_single_qubit_params=0)
    circuit = ansatz(spiders_reader.sentence2diagram(sentence))

    joiner = CX >> (QId(qubit) @ Bra(0))
    expected_circuit = Ket(0,0,0,0) >> joiner @ joiner >> joiner

    assert circuit == expected_circuit


def test_sentence2diagram_bad_tokenised_flag(sentence):
    sentence_tokenised = sentence.split()
    with pytest.raises(ValueError):
        spiders_reader.sentence2diagram(sentence, tokenised=True)
    with pytest.raises(ValueError):
        spiders_reader.sentence2diagram(sentence_tokenised)


def test_dependency_reader(sentence, words, parser):
    with pytest.raises(ValueError):
        DependencyReader(ccg_parser='parser')

    with pytest.raises(ValueError):
        DependencyReader(ccg_parser=lambda: 'parser')

    reader = DependencyReader(ccg_parser=parser,
                              label_relations=False,
                              label_indices=True,
                              label_box_indices=True)

    expected_edges = {(1, 0), (1, 3), (2, 3), (1, 1)}
    expected_boxes = {0: (set(), {(1, 0)}),
                      1: ({(1, 0), (1, 3)}, {(1, 1)}),
                      2: ({(2, 3)}, set()),
                      3: (set(), {(1, 3), (2, 3)})}

    def ty_to_dep(ty):
        toks = ty.name.split()
        return int(toks[0]), int(toks[-1])

    diagram = reader.sentence2diagram(sentence)
    actual_edges = set()
    actual_boxes = {}
    for left, box, right in diagram:
        index = int(box.name[0])
        actual_boxes[index] = ({ty_to_dep(edge) for edge in box.dom},
                               {ty_to_dep(edge) for edge in box.cod})
        actual_edges.update(*actual_boxes[index],
                            (ty_to_dep(edge) for edge in left @ right))

    assert actual_edges == expected_edges
    assert actual_boxes == expected_boxes

def make_parse(*names):
    S = AtomicType.SENTENCE
    boxes = [Box(name, S @ S, S) for name in names]
    return Id(S @ S) @ boxes[0] >> Id(S) @ boxes[1] >> boxes[2]


def test_tree_reader(sentence, words, parser):
    S = AtomicType.SENTENCE
    with pytest.raises(ValueError):
        TreeReader(ccg_parser='parser')

    with pytest.raises(ValueError):
        TreeReader(ccg_parser=lambda: 'parser')

    with pytest.raises(ValueError):
        TreeReader(mode='party mode')

    the_words = Id().tensor(*[Word(w, S) for w in words])

    reader0 = TreeReader(ccg_parser=parser, mode=TreeReaderMode.NO_TYPE)
    mode0_expect = the_words >> make_parse('UNIBOX', 'UNIBOX', 'UNIBOX')
    assert reader0.sentence2diagram(sentence) == mode0_expect

    reader1 = TreeReader(ccg_parser=parser, mode=TreeReaderMode.RULE_ONLY)
    mode1_expect = the_words >> make_parse('FA', 'FA', 'BA')
    assert reader1.sentence2diagram(sentence) == mode1_expect

    reader2 = TreeReader(ccg_parser=parser, mode=TreeReaderMode.RULE_TYPE)
    mode2_expect = the_words >> make_parse('FA(n/n, n)', r'FA((s\n)/n, n)', r'BA(n, s\n)')
    assert reader2.sentence2diagram(sentence) == mode2_expect
    mode2_expect_np = the_words >> make_parse('FA(np/n, n)', r'FA((s\np)/np, np)', r'BA(np, s\np)')
    assert reader2.sentence2diagram(sentence, collapse_noun_phrases=False) == mode2_expect_np

    reader3 = TreeReader(ccg_parser=parser, mode=TreeReaderMode.HEIGHT)
    mode3_expect = the_words >> make_parse('layer_1', 'layer_2', 'layer_3')
    assert reader3.sentence2diagram(sentence) == mode3_expect

def test_suppress_exceptions(sentence):
    bad_parser = WebParser()
    bad_parser.service_url = '..bad..url'

    bad_reader = TreeReader(bad_parser)
    assert bad_reader.sentence2diagram(sentence, suppress_exceptions=True) is None

    bad_reader = TreeReader(bad_parser)
    with pytest.raises(MissingSchema):
        bad_reader.sentence2diagram(sentence, suppress_exceptions=False)


def test_other_readers(sentence):
    # since all the readers share behaviour, just test that they don't fail
    assert stairs_reader.sentence2diagram(sentence)
    assert cups_reader.sentence2diagram(sentence)
