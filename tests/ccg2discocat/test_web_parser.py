import pytest

from discopy import Word
from discopy.rigid import Cup, Diagram, Ty

from lambeq.ccg2discocat.web_parser import WebParser, WebParseError
from lambeq.core.types import AtomicType


@pytest.fixture(scope='module')
def web_parser():
    return WebParser()


def test_sentence2diagram(web_parser):
    sentence = 'he does not sleep'

    n, s = AtomicType.NOUN, AtomicType.SENTENCE
    expected_diagram = Diagram(
        dom=Ty(), cod=Ty('s'),
        boxes=[
            Word('he', n),
            Word('does', n.r @ s @ s.l @ n),
            Word('sleep', n.r @ s),
            Word('not', s.r @ n.r.r @ n.r @ s),
            Cup(s, s.r), Cup(n.r, n.r.r), Cup(n, n.r), Cup(s.l, s), Cup(n, n.r)
        ],
        offsets=[0, 1, 5, 3, 2, 1, 4, 3, 0])

    diagram = web_parser.sentence2diagram(sentence, planar=True)
    assert diagram == expected_diagram


def test_no_exceptions(web_parser):
    assert web_parser.sentences2diagrams(
        [''], suppress_exceptions=True) == [None]

    with pytest.raises(ValueError):
        assert web_parser.sentence2diagram('')


def test_bad_url():
    service_url = "https://cqc.pythonanywhere.com/monoidal/foo"
    bad_parser = WebParser(service_url=service_url)

    assert bad_parser.sentence2diagram(
        "Need a proper url", suppress_exceptions=True) is None
    with pytest.raises(WebParseError):
        bad_parser.sentence2diagram("Need a proper url")
