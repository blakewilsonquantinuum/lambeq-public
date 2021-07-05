"""
Reader
======
A `Reader` is a parser that turn sentences into DisCoPy diagrams, but
not according to the DisCoCat model.

For example, the `LinearReader` combines linearly from left-to-right.

Subclass `Reader` to define a custom reader.

Some simple example readers are included for use:
    box_stairs_reader : LinearReader
        This combines the first two word boxes with a combining box that
        has a single output. Then, each word box is combined with the
        output from the previous combining box to produce a stair
        pattern.
    box_stairs_with_discard_reader : LinearReader
        This is similar to the `box_stairs_reader`, but the combining
        box has two outputs to match the number of inputs, so one output
        is discarded before the remaining output is combined with the
        next word box.
    cups_reader : LinearReader
        This combines each pair of adjacent word boxes with a cup This
        requires each word box to have the output `S >> S` to expose two
        output wires, and a sentinel start box is used to connect to the
        first word box.
    spiders_reader : LinearReader
        This is a special case of the `box_stairs_reader` where the
        combining box is a spider with three legs.

See `examples/readers.ipynb` for illustrative usage.

"""

__all__ = ['Reader', 'LinearReader', 'box_stairs_reader',
           'box_stairs_with_discard_reader', 'cups_reader', 'spiders_reader']

from typing import List, Sequence

from discopy import Word
from discopy.rigid import Box, Cup, Diagram, Id, Ty

from discoket.core.types import AtomicType, Spider

S = AtomicType.SENTENCE


class Reader:
    """Base class for readers.

    This class cannot be used directly, since its methods
    `sentence2diagram` and `sentences2diagrams` call each other. This is
    so that a subclass only needs to implement one method of these
    methods for full functionality (though both can be overridden if
    needed).

    """

    def __new__(cls, *args, **kwargs):
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
                 start_box: Diagram = Id()):
        """Initialise a linear reader.

        Parameters
        ----------
        combining_diagram : Diagram
            The diagram that is used to combine two word boxes. It is
            continuously applied on the left-most wires until a single
            output wire remains.
        word_type : Ty, default: core.types.AtomicType.SENTENCE
            The type of each word box. By default, it uses the sentence
            type from `core.types.AtomicType`.
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


box_stairs_reader = LinearReader(Box('STAIR', S @ S, S))
box_stairs_with_discard_reader = LinearReader(
        Box('STAIR', S @ S, S @ S) >> Box('DISCARD', S, Ty()) @ Id(S))
cups_reader = LinearReader(Cup(S, S.r), S >> S, Word('START', S))
spiders_reader = LinearReader(Spider(2, 1, S))
