"""
Ansatz
======
An ansatz is used to convert a DisCoCat diagram into a quantum circuit.

"""

__all__ = ['Ansatz', 'IQPAnsatz']

from abc import ABC, abstractmethod
from typing import Any, List, Mapping

import discopy
from discoket.core.types import Discard
from discopy.quantum import Discard as QDiscard
from discopy.quantum import qubit
from discopy.quantum.circuit import Functor, Id, IQPansatz as IQP
from discopy.quantum.gates import Bra, Ket, Rx, Rz
from discopy.rigid import Box, Diagram, Ty
import numpy as np
import pytket
from sympy.core.symbol import Symbol


class Ansatz(ABC):
    """Base class for circuit ansatz."""

    def __init__(self, ob_map: Mapping[Ty, int], **kwargs: Any) -> None:
        """Instantiates a circuit ansatz.

        Parameters
        ----------
        ob_map : dict
            A mapping from `discopy.rigid.Ty` to the number of qubits it
            uses in a circuit.
        **kwargs : dict
            Extra parameters for ansatz configuration.

        """
        self.ob_map = ob_map

    @abstractmethod
    def diagram2circuit(self, diagram: Diagram) -> discopy.Circuit:
        """Convert a DisCoPy diagram into a DisCoPy circuit."""

    def diagram2tket(self, diagram: Diagram) -> pytket.Circuit:
        """Convert a DisCoPy diagram into a tket circuit."""
        return self.diagram2circuit(diagram).to_tk()

    def _arity_from_type(self, pg_type: Ty) -> int:
        """Calculate the number of qubits used for a given type."""
        return sum(self.ob_map[Ty(factor.name)] for factor in pg_type)

    def _special_cases(self, ar_map):
        """Convert a discopy box into a tket Circuit element"""
        def new_ar_map(box):
            if isinstance(box, Discard):
                return QDiscard(qubit ** self._arity_from_type(box.dom))
            return ar_map(box)
        return new_ar_map


class IQPAnsatz(Ansatz):
    """Instantaneous Quantum Polynomial anastz.

    An IQP anastz interleaves layers of Hadamard gates with diagonal
    unitaries. This class uses `n-1` adjacent CRz gates to implement
    each diagonal unitary.

    """

    def __init__(self, ob_map: Mapping[Ty, int], n_layers: int,
                 n_single_qubit_params: int) -> None:
        """Instantiates an IQP ansatz.

        Parameters
        ----------
        ob_map : dict
            A mapping from `discopy.rigid.Ty` to the number of qubits it
            uses in a circuit.
        n_layers : int
            The number of IQP layers used by the ansatz.
        n_single_qubit_params : int
            The number of single qubit rotations used by the ansatz.

        """
        super().__init__(ob_map=ob_map, n_layers=n_layers,
                         n_single_qubit_params=n_single_qubit_params)
        self.n_layers = n_layers
        self.n_single_qubit_params = n_single_qubit_params
        self.functor = Functor(ob=self.ob_map,
                               ar=self._special_cases(self._ar))

    def diagram2circuit(self, diagram: Diagram) -> discopy.Circuit:
        return self.functor(diagram)

    def _ar(self, box: Box) -> discopy.Circuit:
        label = self._summarise_box(box)
        dom_arity = self._arity_from_type(box.dom)
        cod_arity = self._arity_from_type(box.cod)

        n_qubits = max(dom_arity, cod_arity)
        n_layers = self.n_layers
        n_single_qubit_params = self.n_single_qubit_params

        if n_qubits == 0:
            circuit = Id()
        elif n_qubits == 1:
            syms = self._make_symbols(label, n_single_qubit_params)
            rots = [Rx, Rz]
            circuit = Id(qubit)
            for i, sym in enumerate(syms):
                circuit >>= rots[i % 2](sym)
        else:
            n_params = n_layers * (n_qubits-1)
            syms = self._make_symbols(label, n_params)
            params = np.array(syms).reshape((n_layers, n_qubits-1))
            circuit = IQP(n_qubits, params)

        if cod_arity <= dom_arity:
            circuit >>= Id(cod_arity) @ Bra(*[0]*(dom_arity - cod_arity))
        else:
            circuit <<= Id(dom_arity) @ Ket(*[0]*(cod_arity - dom_arity))
        return circuit

    @staticmethod
    def _make_symbols(label: str, n_params: int) -> List[Symbol]:
        """Generate a list of symbols for a given label."""
        return [Symbol(f'{label}_{i}') for i in range(n_params)]

    @staticmethod
    def _summarise_box(box: Box) -> str:
        """Summarise the given discopy box."""
        dom = str(box.dom).replace(' @ ', '@')
        cod = str(box.cod).replace(' @ ', '@')
        ret = '_'.join((box.name, dom, cod))
        if ' ' in ret:
            raise ValueError('Name and types of boxes must not contain '
                             f'whitespace (got box with name={box.name},'
                             f' dom={dom}, cod={cod})')
        return ret
