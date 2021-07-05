__all__ = ['CCGParser']

from abc import ABC, abstractmethod
from typing import Any, Iterable, List, Optional

from discopy import Diagram

from discoket.ccg2discocat.ccg_tree import CCGTree


class CCGParser(ABC):
    """Base class for CCG parsers."""

    @abstractmethod
    def __init__(self, **kwargs: Any) -> None:
        """Instantiate the CCG parser."""

    @abstractmethod
    def sentences2trees(
            self,
            sentences: Iterable[str],
            suppress_exceptions: bool = False) -> List[Optional[CCGTree]]:
        """Parse multiple sentences into a list of `CCGTree`s.

        If a sentence fails to parse, its list entry is `None`.
        """

    def sentence2tree(self,
                      sentence: str,
                      suppress_exceptions: bool = False) -> Optional[CCGTree]:
        """Parse a sentence into a `CCGTree`.

        If the sentence fails to parse, it returns `None`.
        """
        return self.sentences2trees([sentence])[0]

    def sentences2diagrams(
            self,
            sentences: Iterable[str],
            planar: bool = False,
            suppress_exceptions: bool = False) -> List[Optional[Diagram]]:
        """Parse multiple sentences into a list of discopy diagrams.

        If a sentence fails to parse, its list entry is `None`.
        """
        diagrams = []
        for tree in self.sentences2trees(sentences):
            if tree is not None:
                try:
                    diagrams.append(tree.to_diagram(planar=planar))
                except Exception as e:
                    if suppress_exceptions:
                        diagrams.append(None)
                    else:
                        raise e
            else:
                diagrams.append(None)
        return diagrams

    def sentence2diagram(
            self,
            sentence: str,
            planar: bool = False,
            suppress_exceptions: bool = False) -> Optional[Diagram]:
        """Parse a sentence into a discopy diagram.

        If the sentence fails to parse, it returns `None`.
        """
        return self.sentences2diagrams([sentence], planar,
                                       suppress_exceptions)[0]
