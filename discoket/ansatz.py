"""
Ansatz
======
An ansatz is used to convert a DisCoCat diagram into a quantum circuit.

"""

__all__ = ['BaseAnsatz', 'Symbol']


from abc import ABC, abstractmethod
from typing import Any, Mapping

from discopy import monoidal, rigid
import sympy


class Symbol(sympy.Symbol):
    """A sympy symbol augmented with extra information.

    Attributes
    ----------
    size : int
        The size of the tensor that this symbol represents.

    """

    def __init__(self, name: str, size: int = 1) -> None:
        """Initialise a symbol.

        Parameters
        ----------
        size : int, default: 1
            The size of the tensor that this symbol represents.

        """
        self._size = size

    @property
    def size(self) -> int:
        return self._size


class BaseAnsatz(ABC):
    """Base class for ansatz."""

    @abstractmethod
    def __init__(self,
                 ob_map: Mapping[rigid.Ty, monoidal.Ty],
                 **kwargs: Any) -> None:
        """Instantiate an ansatz.

        Parameters
        ----------
        ob_map : dict
            A mapping from `discopy.rigid.Ty` to a type in the target
            category. In the category of quantum circuits, this type is
            the number of qubits; in the category of vector spaces, this
            type is a vector space.
        **kwargs : dict
            Extra parameters for ansatz configuration.

        """

    @abstractmethod
    def __call__(self, diagram: rigid.Diagram) -> monoidal.Diagram:
        """Convert a DisCoPy diagram into a DisCoPy circuit or tensor."""

    @staticmethod
    def _summarise_box(box: rigid.Box) -> str:
        """Summarise the given DisCoPy box."""

        dom = str(box.dom).replace(' @ ', '@') if box.dom else ''
        cod = str(box.cod).replace(' @ ', '@') if box.cod else ''
        return f'{box.name}_{dom}_{cod}'
