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
QuantumModel
============
Module containing the base class for a quantum lambeq model.

"""
from __future__ import annotations

import os
import pickle
from abc import abstractmethod
from typing import Union

import numpy as np
from discopy import Tensor
from discopy.tensor import Diagram

from lambeq.training.model import Model


class QuantumModel(Model):
    """Quantum Model base class.

    Attributes
    ----------
    symbols : list of symbols
        A sorted list of all :py:class:`.Symbol`s occurring in the data.
    weights : SizedIterable
        A data structure containing the numeric values of the model
        parameters
    SMOOTHING : float
        A smoothing constant

    """

    SMOOTHING = 1e-9

    def __init__(self) -> None:
        """Initialise an instance of a :py:class:`QuantumModel` base class."""
        super().__init__()

    def _normalise(self, predictions: np.ndarray) -> np.ndarray:
        """Apply smoothing to predictions."""
        backend = Tensor.get_backend()
        predictions = backend.abs(predictions) + self.SMOOTHING
        return predictions / predictions.sum()

    def initialise_weights(self) -> None:
        """Initialise the weights of the model.

        Raises
        ------
        ValueError
            If `model.symbols` are not initialised.

        """
        if not self.symbols:
            raise ValueError('Symbols not initialised. Instantiate through '
                             '`initialise_symbols()`.')
        assert all(w.size == 1 for w in self.symbols)
        self.weights = np.random.rand(len(self.symbols))

    @classmethod
    def load_from_checkpoint(cls,
                             checkpoint_path: Union[str, os.PathLike],
                             **kwargs) -> QuantumModel:
        """Load the model weights and symbols from a training checkpoint.

        Parameters
        ----------
        checkpoint_path : str or PathLike
            Path that points to the checkpoint file.

        Keyword Args
        ------------
        backend_config : dict
            Dictionary containing the backend configuration for the
            :py:class:`TketModel`. Must include the fields `'backend'`,
            `'compilation'` and `'shots'`.

        Raises
        ------
        FileNotFoundError
            If checkpoint file does not exist.

        """
        model = cls(**kwargs)
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, 'rb') as ckp:
                checkpoint = pickle.load(ckp)
            try:
                model.symbols = checkpoint['model_symbols']
                model.weights = checkpoint['model_weights']
                return model
            except KeyError as e:
                raise e
        else:
            raise FileNotFoundError('Checkpoint not found! Check path '
                                    f'{checkpoint_path}')

    @abstractmethod
    def get_diagram_output(self, diagrams: list[Diagram]) -> np.ndarray:
        """Return the diagram prediction.
        Parameters
        ----------
        diagrams : list of diagram
            List of lambeq diagrams.
        """

    @abstractmethod
    def forward(self, x: list[Diagram]) -> np.ndarray:
        """The forward pass of the model."""
