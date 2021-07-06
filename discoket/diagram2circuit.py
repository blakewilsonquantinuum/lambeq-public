"""
Ansatz
======
An ansatz is used to convert a DisCoCat diagram into a quantum circuit.

"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from discopy.rigid import Box, Diagram, Ty
from discopy.quantum import Bra, Ket
from discopy.quantum import Circuit as DiscopyCircuit
from discopy.quantum import IQPansatz as IQP
from discopy.quantum.circuit import Functor, Id
import numpy as np
from pytket import Circuit as TketCircuit
from sympy.abc import symbols
from sympy.core.symbol import Symbol

__all__ = ['Ansatz', 'IQPAnsatz']


def _summarise_box(box: Box) -> str:
    """Summarise the given discopy box."""
    dom = str(box.dom).replace(' @ ', '@')
    cod = str(box.cod).replace(' @ ', '@')
    args = box.name, dom, cod
    if any([' ' in arg for arg in args]):
        exc = 'Name and types of boxes should not contain whitespace'
        raise ValueError(exc)

    return '_'.join(args)


def _make_symbols(label: str, n_params: int) -> List[Symbol]:
    """Generate a list of symbols for a given label."""
    return [symbols(f"{label}_{i}") for i in range(n_params)]


class Ansatz(ABC):
    """Base class for circuit ansatz."""

    @abstractmethod
    def __init__(self, ob_map: Dict[Ty, int], **kwargs: Any) -> None:
        """Instantiates a circuit ansatz.

        Parameters
        ----------
        ob_map : dict
            A dictionary which maps `discopy.rigid.Ty` into an integer, the
            number of qubits it has in circuit form.
        kwargs : dict
            Extra parameters for ansatz configuration.

        """

    def __call__(self, diagram: Diagram) -> DiscopyCircuit:
        """Convert the given discopy diagram into a discopy circuit."""
        return self._functor(diagram)

    def diagram2tket(self, diagram: Diagram) -> TketCircuit:
        """Convert the given discopy diagram into a tket circuit."""
        return self(diagram).to_tk()

    def _arity_from_type(self, pg_type: Ty) -> int:
        """Calculate number of circuit qubits for the given pregroup type."""
        return sum(self._ob[Ty(factor.name)] for factor in pg_type)


class IQPAnsatz(Ansatz):
    """Class which implements the IQP ansatz.

    A Instantaneous Quantum Polynomial (IQP) anastz involves interleaving
    layers of Hadamrd gates with diagonal unitaries. This class uses n-1
    adjacent CRz gates to implement each diagonal unitary.

    """

    def __init__(self, ob_map: Dict[Ty, int], n_layers: int) -> None:
        """Instantiates an IQP ansatz.

        Parameters
        ----------
        ob_map : dict
            A dictionary which maps `discopy.rigid.Ty` into an integer, the
            number of qubits it has in circuit form.
        n_layers : int
            The number of IQP layers used by the ansatz.

        """
        self._ob = ob_map
        self.n_layers = n_layers

        self._functor = Functor(ob=self._ob, ar=self._ar)

    def _ar(self, box: Box) -> DiscopyCircuit:
        label = _summarise_box(box)
        dom_arity = self._arity_from_type(box.dom)
        cod_arity = self._arity_from_type(box.cod)

        n_qubits = max(dom_arity, cod_arity)
        n_layers = self.n_layers

        if n_qubits == 0:
            circuit = Id()
        elif n_qubits == 1:
            # Rx(v0) >> Rz(v1) >> Rx(v2)
            circuit = IQP(1, _make_symbols(label, 3))
        else:
            n_params = n_layers * (n_qubits-1)
            syms = _make_symbols(label, n_params)
            params = np.array(syms).reshape((n_layers, n_qubits-1))
            circuit = IQP(n_qubits, params)

        if cod_arity <= dom_arity:
            circuit >>= Id(cod_arity) @ Bra(*[0]*(dom_arity - cod_arity))
        else:
            circuit <<= Id(dom_arity) @ Ket(*[0]*(cod_arity - dom_arity))
        return circuit
