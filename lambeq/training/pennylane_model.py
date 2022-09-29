# Copyright 2021-2022 Cambridge Quantum Computing Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
PennyLaneModel
==============
Module implementing quantum and quantum/classical hybrid lambeq models,
based on a PennyLane and PyTorch backend.

"""
from __future__ import annotations

import copy
from typing import Any, Optional, TYPE_CHECKING

from discopy import Circuit, Diagram
from sympy import default_sort_key
import torch

from lambeq.training.checkpoint import Checkpoint
from lambeq.training.pytorch_model import PytorchModel

if TYPE_CHECKING:
    import pennylane as qml
    from discopy.quantum.pennylane import PennyLaneCircuit


def _import_pennylane() -> None:
    global qml
    import pennylane as qml


STATE_BACKENDS = ['default.qubit', 'lightning.qubit', 'qiskit.aer']
STATE_DEVICES = ['aer_simulator_statevector', 'statevector_simulator']


class PennyLaneModel(PytorchModel):
    """ A lambeq model for the quantum and hybrid quantum/classical
    pipeline using PennyLane circuits. This model inherits from the
    PytorchModel because PennyLane interfaces with PyTorch to allow
    backpropagation through hybrid models.

    """

    def __init__(self,
                 backend_config: Optional[dict[str, Any]] = None) -> None:
        """Initialise a :py:class:`PennyLaneModel` instance with
        an empty `circuit_map` dictionary.

        Parameters
        ----------
        probabilities : bool, default: True
            Whether to use probabilities or states for the output.
        backend_config : dict, optional
            Configuration for hardware or simulator to be used. Defaults
            to using the `default.qubit` PennyLane simulator analytically,
            with normalized probability outputs. Keys that can be used
            include 'backend', 'device', 'probabilities', 'normalize',
            'shots', and 'noise_model'.

        """
        super().__init__()
        _import_pennylane()
        self.circuit_map: dict[Circuit, PennyLaneCircuit] = {}

        if backend_config is None:
            backend_config = {'backend': 'default.qubit'}

        backend_config = {'probabilities': True,
                          'normalize': True,
                          **backend_config}

        self._backend = backend_config.pop('backend')

        if self._backend == 'honeywell.hqs':
            try:
                backend_config['machine'] = backend_config.pop('device')
            except KeyError:
                raise ValueError('When using the honeywell.hqs provider, '
                                 'a device must be specified.')
        elif 'device' in backend_config:
            backend_config['backend'] = backend_config.pop('device')

        self._probabilities = backend_config.pop('probabilities')
        self._normalize = backend_config.pop('normalize')
        self._backend_config = backend_config

        if not self._probabilities:
            if self._backend not in STATE_BACKENDS:
                raise ValueError(f'The {self._backend} backend is not '
                                 'compatible with state outputs.')
            elif ('backend' in self._backend_config
                  and self._backend_config['backend'] not in STATE_DEVICES):
                raise ValueError(f'The {self._backend_config["backend"]} '
                                 'device is not compatible with state '
                                 'outputs.')

    def _load_checkpoint(self, checkpoint: Checkpoint) -> None:
        """Load the model weights and symbols from a lambeq
        :py:class:`.Checkpoint`.

        Parameters
        ----------
        checkpoint : :py:class:`.Checkpoint`
            Checkpoint containing the model weights,
            symbols and additional information.

        """
        self.symbols = checkpoint['model_symbols']
        self.weights = checkpoint['model_weights']
        self._probabilities = checkpoint['model_probabilities']
        self._normalize = checkpoint['model_normalize']
        self._backend = checkpoint['model_backend']
        self._backend_config = checkpoint['model_backend_config']
        self.circuit_map = checkpoint['model_circuits']
        self.load_state_dict(checkpoint['model_state_dict'])

        for p_circ in self.circuit_map.values():
            p_circ.device = qml.device(self._backend,
                                       wires=p_circ.n_qubits,
                                       **self._backend_config)

    def _make_checkpoint(self) -> Checkpoint:
        """Create checkpoint that contains the model weights and symbols.

        Returns
        -------
        :py:class:`.Checkpoint`
            Checkpoint containing the model weights, symbols and
            additional information.

        """

        checkpoint = Checkpoint()
        circuit_map = {k: copy.copy(v) for k, v in self.circuit_map.items()}
        for c in circuit_map.values():
            c.device = None

        checkpoint.add_many({'model_weights': self.weights,
                             'model_symbols': self.symbols,
                             'model_probabilities': self._probabilities,
                             'model_normalize': self._normalize,
                             'model_backend': self._backend,
                             'model_backend_config': self._backend_config,
                             'model_circuits': circuit_map,
                             'model_state_dict': self.state_dict()})

        return checkpoint

    def initialise_weights(self) -> None:
        """Initialise the weights of the model.

        Raises
        ------
        ValueError
            If `model.symbols` are not initialised.

        """
        self._reinitialise_modules()
        if not self.symbols:
            raise ValueError('Symbols not initialised. Instantiate through '
                             '`PytorchModel.from_diagrams()`.')
        self.weights = torch.nn.ParameterList([torch.rand(w.size)
                                               for w in self.symbols])

    def get_diagram_output(self, diagrams: list[Diagram]) -> torch.Tensor:
        """Evaluate outputs of circuits using PennyLane.

        Parameters
        ----------
        diagrams : list of :py:class:`~discopy.tensor.Diagram`
            The :py:class:`Diagrams <discopy.tensor.Diagram>` to be
            evaluated.

        Raises
        ------
        ValueError
            If `model.weights` or `model.symbols` are not initialised.

        Returns
        -------
        torch.Tensor
            Resulting tensor.

        """
        circuit_evals = [self.circuit_map[d].eval(self.symbols, self.weights)
                         for d in diagrams]
        if self._normalize:
            if self._probabilities:
                circuit_evals = [c / torch.sum(c) for c in circuit_evals]
            else:
                circuit_evals = [c / torch.sum(torch.square(torch.abs(c)))
                                 for c in circuit_evals]

        stacked = torch.stack(circuit_evals).squeeze(-1)
        if self._probabilities:
            return stacked.to(self.weights[0].dtype)
        else:
            return stacked

    def forward(self, x: list[Diagram]) -> torch.Tensor:
        """Perform default forward pass by running circuits.

        In case of a different datapoint (e.g. list of tuple) or
        additional computational steps, please override this method.

        Parameters
        ----------
        x : list of :py:class:`~discopy.quantum.Circuit`
            The :py:class:`Circuits <discopy.quantum.Circuit>` to be
            evaluated.

        Returns
        -------
        torch.Tensor
            Tensor containing model's prediction.

        """
        return self.get_diagram_output(x)

    @classmethod
    def from_diagrams(cls,
                      diagrams: list[Diagram],
                      backend_config: Optional[dict[str, Any]] = None,
                      **kwargs: Any) -> PennyLaneModel:
        """Build model from a list of
        :py:class:`Circuits <discopy.quantum.Circuit>`.

        Parameters
        ----------
        diagrams : list of :py:class:`~discopy.quantum.Circuit`
            The circuit diagrams to be evaluated.
        backend_config : dict, optional
            Configuration for hardware or simulator to be used. Defaults
            to using the `default.qubit` PennyLane simulator analytically,
            with normalized probability outputs. Keys that can be used
            include 'backend', 'device', 'probabilities', 'normalize',
            'shots', and 'noise_model'.

        """
        if not all(isinstance(x, Circuit) for x in diagrams):
            raise ValueError('All diagrams must be of type'
                             '`discopy.quantum.Circuit`.')

        model = cls(backend_config=backend_config, **kwargs)

        model.symbols = sorted(
            {sym for circ in diagrams for sym in circ.free_symbols},
            key=default_sort_key)
        for circ in diagrams:
            p_circ = circ.to_pennylane(probabilities=model._probabilities)
            p_circ.device = qml.device(model._backend,
                                       wires=p_circ.n_qubits,
                                       **model._backend_config)

            model.circuit_map[circ] = p_circ

        return model
