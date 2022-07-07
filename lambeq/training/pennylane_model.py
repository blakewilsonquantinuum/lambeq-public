# Copyright 2021, 2022 Cambridge Quantum Computing Ltd.
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
import os
import pickle
from typing import Any, Union

from sympy import default_sort_key
import torch

import discopy
from discopy import Diagram, Circuit
from lambeq.training.checkpoint import Checkpoint
from lambeq.training.pytorch_model import PytorchModel


class PennyLaneModel(PytorchModel):
    """ A lambeq model for the quantum and hybrid quantum/classical
    pipeline using PennyLane circuits. This model inherits from the
    PytorchModel because PennyLane interfaces with PyTorch to allow
    backpropagation through hybrid models.

    """

    def __init__(self) -> None:
        """Initialises a :py:class:`PennyLaneModel` instance with
        an empty `circuit_map` dictionary.

        """
        PytorchModel.__init__(self)
        self.circuit_map: dict[Circuit,
                               discopy.quantum.pennylane.PennyLaneCircuit] = {}

    @classmethod
    def from_checkpoint(cls,
                        checkpoint_path: Union[str, os.PathLike],
                        **kwargs) -> PennyLaneModel:
        """Load the model's weights, symbols, and circuits from a training
        checkpoint.

        Parameters
        ----------
        checkpoint_path : str or `os.PathLike`
            Path that points to the checkpoint file.

        Raises
        ------
        FileNotFoundError
            If checkpoint file does not exist.

        """
        model = cls(**kwargs)
        checkpoint = Checkpoint.from_file(checkpoint_path)
        try:
            model.symbols = checkpoint['model_symbols']
            model.weights = checkpoint['model_weights']
            model.circuit_map = checkpoint['model_circuits']
            model.load_state_dict(checkpoint['model_state_dict'])
            return model
        except KeyError as e:
            raise e

    def make_checkpoint(self,
                        checkpoint_path: Union[str, os.PathLike]) -> None:
        """Make a checkpoint file for the model.

        Parameters
        ----------
        checkpoint_path : str or `os.PathLike`
            Path that points to the checkpoint file. If
            the file does not exist, it will be created.

        Raises
        ------
        FileNotFoundError
            If the directory in which the checkpoint file is to be
            saved does not exist.

        """
        checkpoint = Checkpoint()
        checkpoint.add_many({'model_weights': self.weights,
                             'model_symbols': self.symbols,
                             'model_circuits': self.circuit_map,
                             'model_state_dict': self.state_dict()})

        checkpoint.to_file(checkpoint_path)

    def get_diagram_output(self, diagrams: list[Diagram]) -> torch.Tensor:
        """Evaluate outputs of circuits using PennyLane.

        Parameters
        ----------
        diagrams : list of :py:class:`~discopy.tensor.Diagram`
            The :py:class:`Diagrams <discopy.tensor.Diagram>` to be evaluated.

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

        return torch.stack(circuit_evals)

    def forward(self, x: list[Diagram]) -> torch.Tensor:
        """Perform default forward pass of a lambeq model by
        running circuits.

        In case of a different datapoint (e.g. list of tuple) or additional
        computational steps, please override this method.

        Parameters
        ----------
        x : list of :py:class:`~discopy.quantum.Circuit`
            The :py:class:`Circuits <discopy.quantum.Circuit>` to be evaluated.

        Returns
        -------
        torch.Tensor
            Tensor containing model's prediction.

        """
        return self.get_diagram_output(x)

    @classmethod
    def from_diagrams(cls, diagrams: list[Diagram],
                      probabilities=True,
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

        """
        if not all(isinstance(x, Circuit) for x in diagrams):
            raise ValueError("All diagrams must be of type"
                             "`discopy.quantum.Circuit`.")

        model = cls(**kwargs)
        model.symbols = sorted(
            {sym for circ in diagrams for sym in circ.free_symbols},
            key=default_sort_key)
        model.circuit_map = {circ:
                             circ.to_pennylane(probabilities=probabilities)
                             for circ in diagrams}
        return model
