import pytest

from discopy import Box, Cup, Ty, Word
from discopy.quantum import Bra, CRz, CX, H, Ket, Rx, Rz, sqrt
from discopy.quantum.circuit import Id
from discoket.core.types import AtomicType
from discoket.diagram2circuit import IQPAnsatz
from pytket.qasm import circuit_to_qasm_str
from sympy.abc import symbols as sym

N = AtomicType.NOUN
S = AtomicType.SENTENCE


def test_iqp_ansatz():
    diagram = (Word('Alice', N) @ Word('runs', N >> S) >>
               Cup(N, N.r) @ Id(S))
    ansatz = IQPAnsatz({N: 1, S: 1}, n_layers=1)

    expected_circuit = (Ket(0) >>
                        Rx(sym('Alice_Ty()_n_0')) >>
                        Rz(sym('Alice_Ty()_n_1')) >>
                        Rx(sym('Alice_Ty()_n_2')) >>
                        Id(1) @ Ket(0, 0) >> Id(1) @ H @ Id(1) >>
                        Id(2) @ H >>
                        Id(1) @ CRz(sym('runs_Ty()_n.r@s_0')) >>
                        CX @ Id(1) >>
                        H @ Id(2) >>
                        Id(1) @ sqrt(2) @ Id(2) >>
                        Bra(0, 0) @ Id(1))
    assert ansatz.diagram2circuit(diagram) == expected_circuit

    expected_qasm = """OPENQASM 2.0;
include "qelib1.inc";

qreg q[3];
creg c[2];
rx((2*Alice_Ty()_n_0)*pi) q[0];
h q[1];
h q[2];
rz((2*Alice_Ty()_n_1)*pi) q[0];
crz((2*runs_Ty()_n.r@s_0)*pi) q[1],q[2];
rx((2*Alice_Ty()_n_2)*pi) q[0];
cx q[0],q[1];
measure q[1] -> c[1];
h q[0];
measure q[0] -> c[0];
"""

    assert circuit_to_qasm_str(ansatz.diagram2tket(diagram)) == expected_qasm


def test_bad_summary():
    d = Word("no spaces allowed", S)
    ansatz = IQPAnsatz({N: 0, S: 1}, n_layers=1)
    with pytest.raises(ValueError):
        ansatz.diagram2circuit(d)


def test_iqp_ansatz_inverted():
    d = Box("inverted", S, Ty())
    ansatz = IQPAnsatz({N: 0, S: 0}, n_layers=1)
    assert ansatz.diagram2circuit(d) == Bra()


def test_iqp_ansatz_empty():
    diagram = (Word('Alice', N) @ Word('runs', N >> S) >>
               Cup(N, N.r) @ Id(S))
    ansatz = IQPAnsatz({N: 0, S: 0}, n_layers=1)
    assert ansatz.diagram2circuit(diagram) == Bra() >> Bra()
