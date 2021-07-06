from discopy import Box, Cup, Diagram, Ty, Word
from discopy.quantum import Bra, CRz, CX, H, Ket, Rx, Rz, sqrt
from discopy.quantum.circuit import Id
from discoket.core.types import AtomicType
from discoket.diagram2circuit import IQPAnsatz, _summarise_box
import pytest
from pytket.qasm import circuit_to_qasm_str
from sympy.abc import symbols as sym

N = AtomicType.NOUN
S = AtomicType.SENTENCE


def test_iqp_ansatz():
    d = Diagram(
        dom=Ty(),
        cod=S,
        boxes=[
            Word('Alice', N),
            Word('runs', N.r @ S),
            Cup(N, N.r)],
        offsets=[0, 1, 0]
    )

    ansatz = IQPAnsatz({N: 1, S: 1}, n_layers=1)

    discopy_circuit = ansatz(d)
    tket_circuit = ansatz.diagram2tket(d)

    expected_circuit = (
        Ket(0) >> Rx(sym('Alice_Ty()_n_0')) >> Rz(sym('Alice_Ty()_n_1')) >>
        Rx(sym('Alice_Ty()_n_2')) >> Id(1) @ Ket(0, 0) >> Id(1) @ H @ Id(1) >>
        Id(2) @ H >> Id(1) @ CRz(sym('runs_Ty()_n.r@s_0')) >> CX @ Id(1) >>
        H @ Id(2) >> Id(1) @ sqrt(2) @ Id(2) >> Bra(0, 0) @ Id(1)
    )

    expected_qasm = (
        'OPENQASM 2.0;\n'
        'include "qelib1.inc";\n'
        '\n'
        'qreg q[3];\n'
        'creg c[2];\n'
        'rx((2*Alice_Ty()_n_0)*pi) q[0];\n'
        'h q[1];\n'
        'h q[2];\n'
        'rz((2*Alice_Ty()_n_1)*pi) q[0];\n'
        'crz((2*runs_Ty()_n.r@s_0)*pi) q[1],q[2];\n'
        'rx((2*Alice_Ty()_n_2)*pi) q[0];\n'
        'cx q[0],q[1];\n'
        'measure q[1] -> c[1];\n'
        'h q[0];\n'
        'measure q[0] -> c[0];\n'
    )

    assert discopy_circuit == expected_circuit
    assert circuit_to_qasm_str(tket_circuit) == expected_qasm


def test_bad_summary():
    d = Word("no spaces allowed", S)
    ansatz = IQPAnsatz({N: 0, S: 1}, n_layers=1)
    with pytest.raises(ValueError):
        ansatz(d)


def test_iqp_ansatz_inverted():
    d = Box("inverted", S, Ty())
    ansatz = IQPAnsatz({N: 0, S: 0}, n_layers=1)

    assert ansatz(d) == Bra()


def test_iqp_ansatz_empty():
    d = Diagram(
        dom=Ty(),
        cod=S,
        boxes=[
            Word('Alice', N),
            Word('runs', N.r @ S),
            Cup(N, N.r)],
        offsets=[0, 1, 0]
    )

    from discopy.quantum.circuit import Id
    ansatz = IQPAnsatz({N: 0, S: 0}, n_layers=1)
    assert ansatz(d) == Bra() >> Bra()
