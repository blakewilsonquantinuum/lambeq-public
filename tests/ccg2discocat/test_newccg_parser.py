from io import StringIO
import pytest
from unittest.mock import patch

from lambeq import CCGAtomicType, NewCCGParser
from lambeq.core.globals import VerbosityLevel


@pytest.fixture(scope='module')
def newccg_parser():
    return NewCCGParser(verbose=VerbosityLevel.SUPPRESS.value)

@pytest.fixture
def sentence():
    return 'What Alice is and is not .'

@pytest.fixture
def tokenised_sentence():
    return ['What', 'Alice', 'is', 'and', 'is', 'not', '.']

def test_sentence2diagram(newccg_parser, sentence):
    assert newccg_parser.sentence2diagram(sentence) is not None


def test_sentence2tree(newccg_parser, sentence):
    assert newccg_parser.sentence2tree(sentence) is not None


def test_sentence2tree_tokenised(newccg_parser, tokenised_sentence):
    assert newccg_parser.sentence2tree(tokenised_sentence, tokenised=True) is not None


def test_sentences2diagrams(newccg_parser, sentence):
    assert newccg_parser.sentences2diagrams([sentence]) is not None


def test_sentence2diagram_tokenised(newccg_parser, tokenised_sentence):
    assert newccg_parser.sentence2diagram(tokenised_sentence, tokenised=True) is not None


def test_sentences2diagrams_tokenised(newccg_parser, tokenised_sentence):
    tokenised_sentence = ['What', 'Alice', 'is', 'and', 'is', 'not', '.']
    assert newccg_parser.sentences2diagrams([tokenised_sentence], tokenised=True) is not None


def test_tokenised_type_check_untokenised_sentence(newccg_parser, sentence):
    with pytest.raises(ValueError):
        _=newccg_parser.sentence2diagram(sentence, tokenised=True)


def test_tokenised_type_check_tokenised_sentence(newccg_parser, tokenised_sentence):
    with pytest.raises(ValueError):
        _=newccg_parser.sentence2diagram(tokenised_sentence, tokenised=False)


def test_tokenised_type_check_untokenised_batch(newccg_parser, sentence):
    with pytest.raises(ValueError):
        _=newccg_parser.sentences2diagrams([sentence], tokenised=True)


def test_tokenised_type_check_tokenised_batch(newccg_parser, tokenised_sentence):
    with pytest.raises(ValueError):
        _=newccg_parser.sentences2diagrams([tokenised_sentence], tokenised=False)


def test_tokenised_type_check_untokenised_sentence_s2t(newccg_parser, sentence):
    with pytest.raises(ValueError):
        _=newccg_parser.sentence2tree(sentence, tokenised=True)


def test_tokenised_type_check_tokenised_sentence_s2t(newccg_parser, tokenised_sentence):
    with pytest.raises(ValueError):
        _=newccg_parser.sentence2tree(tokenised_sentence, tokenised=False)


def test_tokenised_type_check_untokenised_batch_s2t(newccg_parser, sentence):
    with pytest.raises(ValueError):
        _=newccg_parser.sentences2trees([sentence], tokenised=True)


def test_tokenised_type_check_tokenised_batch_s2t(newccg_parser, tokenised_sentence):
    with pytest.raises(ValueError):
        _=newccg_parser.sentences2trees([tokenised_sentence], tokenised=False)


def test_verbosity_exceptions_init():
    with pytest.raises(ValueError):
        newccgbank_parser = NewCCGParser(verbose='invalid_option')


def test_kwargs_exceptions_init():
    with pytest.raises(TypeError):
        newccgbank_parser = NewCCGParser(nonexisting_arg='invalid_option')


def test_verbosity_exceptions_sentences2trees(newccg_parser, sentence):
    with pytest.raises(ValueError):
        _=newccg_parser.sentences2trees([sentence], verbose='invalid_option')


def test_text_progress(newccg_parser, sentence):
    with patch('sys.stderr', new=StringIO()) as fake_out:
        _=newccg_parser.sentences2diagrams([sentence], verbose=VerbosityLevel.TEXT.value)
        assert fake_out.getvalue().rstrip() == 'Tagging sentences.\nParsing tagged sentences.\nTurning parse trees to diagrams.'


def test_tqdm_progress(newccg_parser, sentence):
    with patch('sys.stderr', new=StringIO()) as fake_out:
        _=newccg_parser.sentences2diagrams([sentence], verbose=VerbosityLevel.TEXT.value)
        assert fake_out.getvalue().rstrip() != ''


def test_root_filtering(newccg_parser):
    S = CCGAtomicType.SENTENCE
    N = CCGAtomicType.NOUN

    restricted_parser = NewCCGParser(root_cats=['NP'])

    sentence1 = 'do'
    assert newccg_parser.sentence2tree(sentence1).biclosed_type == N >> S
    assert restricted_parser.sentence2tree(sentence1).biclosed_type == N

    sentence2 = 'I do'
    assert newccg_parser.sentence2tree(sentence2).biclosed_type == S
    assert restricted_parser.sentence2tree(sentence2).biclosed_type == N
