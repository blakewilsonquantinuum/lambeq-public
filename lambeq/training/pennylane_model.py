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

import discopy
from discopy import Circuit, Diagram
from sympy import default_sort_key
import torch

from lambeq.training.checkpoint import Checkpoint
from lambeq.training.pytorch_model import PytorchModel


class PennyLaneModel(PytorchModel):
    """ A lambeq model for the quantum and hybrid quantum/classical
    pipeline using PennyLane circuits. This model inherits from the
    PytorchModel because PennyLane interfaces with PyTorch to allow
    backpropagation through hybrid models.

    """

    def __init__(self, probabilities=True, normalize=True) -> None:
        """Initialise a :py:class:`PennyLaneModel` instance with
        an empty `circuit_map` dictionary.

        Parameters
        ----------
        probabilities : bool, default: True
            Whether to use probabilities or states for the output.
        normalize : bool, default: True
            Whether to normalize the output after post-selection.

        """
        PytorchModel.__init__(self)
        self.circuit_map: dict[Circuit,
                               discopy.quantum.pennylane.PennyLaneCircuit] = {}
        self._probabilities = probabilities
        self._normalize = normalize

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
        self.circuit_map = checkpoint['model_circuits']
        self.load_state_dict(checkpoint['model_state_dict'])

    def _make_checkpoint(self) -> Checkpoint:
        """Create checkpoint that contains the model weights and symbols.

        Returns
        -------
        :py:class:`.Checkpoint`
            Checkpoint containing the model weights, symbols and
            additional information.

        """

        checkpoint = Checkpoint()
        checkpoint.add_many({'model_weights': self.weights,
                             'model_symbols': self.symbols,
                             'model_circuits': self.circuit_map,
                             'model_state_dict': self.state_dict()})

        return checkpoint

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

        return torch.stack(circuit_evals)

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
    def from_diagrams(cls, diagrams: list[Diagram],
                      probabilities=True,
                      normalize=True,
                      **kwargs) -> PennyLaneModel:
        """Build model from a list of
        :py:class:`Circuits <discopy.quantum.Circuit>`.

        Parameters
        ----------
        diagrams : list of :py:class:`~discopy.quantum.Circuit`
            The circuit diagrams to be evaluated.
        probabilities : bool, default: True
            Whether the circuits return normalized probabilities or
            unnormalized states in the computational basis.
        normalize : bool, default: True
            Whether to normalize the outputs of the circuits.
            For probabilities, this means the sum of the output tensor
            is 1, while for states it means the sum of the squares of
            the absolute values of the tensor is 1.

        """
        if not all(isinstance(x, Circuit) for x in diagrams):
            raise ValueError('All diagrams must be of type'
                             '`discopy.quantum.Circuit`.')

        model = cls(probabilities=probabilities, normalize=normalize, **kwargs)
        model.symbols = sorted(
            {sym for circ in diagrams for sym in circ.free_symbols},
            key=default_sort_key)
        model.circuit_map = {circ:
                             circ.to_pennylane(probabilities=probabilities)
                             for circ in diagrams}

        return model
