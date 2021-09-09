"""
Reader
======
A :py:class:`Reader` is a parser that turns sentences into DisCoPy
diagrams, but not according to the DisCoCat model.

For example, the :py:class:`LinearReader` combines linearly from
left-to-right.

Subclass :py:class:`Reader` to define a custom reader.

Some simple example readers are included for use:
    :py:data:`cups_reader` : :py:class:`LinearReader`
        This combines each pair of adjacent word boxes with a cup. This
        requires each word box to have the output :py:obj:`S >> S` to
        expose two output wires, and a sentinel start box is used to
        connect to the first word box.
    :py:data:`spiders_reader` : :py:class:`LinearReader`
        This combines the first two word boxes using a spider with three
        legs. The remaining output is combined with the next word box
        using another spider, and so on, until a single output remains.
        Here, each word box has an output type of :py:obj:`S @ S`.

See `examples/readers.ipynb` for illustrative usage.

"""

from __future__ import annotations

__all__ = ['Reader', 'LinearReader', 'cups_reader', 'spiders_reader']

from typing import Any, List, Sequence

from discopy import Word
from discopy.rigid import Cup, Diagram, Id, Spider, Ty

from discoket.core.types import AtomicType, Discard

S = AtomicType.SENTENCE
DISCARD = Discard(S)


class Reader:
    """Base class for readers.

    This class cannot be used directly, since its methods
    :py:meth:`sentence2diagram` and :py:meth:`sentences2diagrams` call
    each other. This is so that a subclass only needs to implement one
    method of these methods for full functionality (though both can be
    overridden if needed).

    """

    def __new__(cls, *args: Any, **kwargs: Any) -> Reader:
        if (cls.sentence2diagram == Reader.sentence2diagram and
                cls.sentences2diagrams == Reader.sentences2diagrams):
            raise TypeError(
                    'This class cannot be directly instantiated since neither '
                    '`sentence2diagram` and `sentences2diagrams` have been '
                    'implemented. See `help(Reader)` for more details.')
        return super().__new__(cls)

    def sentence2diagram(self, sentence: str) -> Diagram:
        """Parse a sentence into a DisCoPy diagram."""
        return self.sentences2diagrams([sentence])[0]

    def sentences2diagrams(self, sentences: Sequence[str]) -> List[Diagram]:
        """Parse multiple sentences into a list of DisCoPy diagrams."""
        return [self.sentence2diagram(sentence) for sentence in sentences]


class LinearReader(Reader):
    """A reader that combines words linearly using a stair diagram."""

    def __init__(self,
                 combining_diagram: Diagram,
                 word_type: Ty = S,
                 start_box: Diagram = Id()) -> None:
        """Initialise a linear reader.

        Parameters
        ----------
        combining_diagram : Diagram
            The diagram that is used to combine two word boxes. It is
            continuously applied on the left-most wires until a single
            output wire remains.
        word_type : Ty, default: core.types.AtomicType.SENTENCE
            The type of each word box. By default, it uses the sentence
            type from :py:class:`.core.types.AtomicType`.
        start_box : Diagram, default: Id()
            The start box used as a sentinel value for combining. By
            default, the empty diagram is used.

        """

        self.combining_diagram = combining_diagram
        self.word_type = word_type
        self.start_box = start_box

    def sentence2diagram(self, sentence: str) -> Diagram:
        """Parse a sentence into a DisCoPy diagram.

        This splits the sentence into words by whitespace, creates a
        box for each word, and combines them linearly.

        """
        words = (Word(word, self.word_type) for word in sentence.split())
        diagram = Diagram.tensor(self.start_box, *words)
        while len(diagram.cod) > 1:
            diagram >>= (self.combining_diagram @
                         Id(diagram.cod[len(self.combining_diagram.dom):]))
        return diagram


cups_reader = LinearReader(Cup(S, S.r), S >> S, Word('START', S))
spiders_reader = LinearReader(Spider(2, 1, S))
