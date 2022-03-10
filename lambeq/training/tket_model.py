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
TketModel
=========
Module implementing a lambeq model based on a quantum backend, via `tket`.

"""
from __future__ import annotations

import os
import pickle
from typing import Any, Callable, Union

import numpy as np
from discopy.quantum import Circuit, Id, Measure
from discopy.tensor import Diagram, Tensor

from lambeq.training.model import Model


class TketModel(Model):
    """A lambeq model for either shot-based simulations of a quantum
    pipeline or experiments run on quantum hardware using `tket`."""

    SMOOTHING = 1e-9

    def __init__(self, **kwargs) -> None:
        """Initialise TketModel based on the `t|ket>` backend.

        Keyword Args
        ------------
        backend_config : dict
            Dictionary containing the backend configuration. Must include the
            fields `'backend'`, `'compilation'` and `'shots'`.

        Raises
        ------
        KeyError
            If `backend_config` is not provided or `backend_config` has missing
            fields.

        """
        if not 'backend_config' in kwargs:
            raise KeyError('Please provide a backend configuration.')

        super().__init__()
        Tensor.np = np

        backend_config = kwargs['backend_config']
        fields = ('backend', 'compilation', 'shots')
        missing_fields = [f for f in fields if f not in backend_config]
        if missing_fields:
            raise KeyError('Missing arguments in backend configuation. '
                           f'Missing arguments: {missing_fields}.')
        self.backend_config = backend_config

    def _make_lambda(self, diagram: Diagram) -> Callable[[Any], Any]:
        """Measure and lambdify diagrams."""
        measured = diagram >> Id().tensor(*[Measure()] * len(diagram.cod))
        return measured.lambdify(*self.symbols)

    def _randint(self, low=-1 << 63, high=(1 << 63)-1):
        return np.random.randint(low, high)

    def _normalise(self, predictions: np.ndarray) -> np.ndarray:
        """Apply smoothing to predictions."""
        predictions = np.abs(predictions) + self.SMOOTHING
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
                             '`TketModel.initialise_symbols()`.')
        assert all(w.size == 1 for w in self.symbols)
        self.weights = np.random.rand(len(self.symbols))

    @classmethod
    def load_from_checkpoint(cls,
                             checkpoint_path: Union[str, os.PathLike],
                             **kwargs) -> TketModel:
        """Load the model weights and symbols from a training checkpoint.

        Parameters
        ----------
        checkpoint_path : str or PathLike
            Path that points to the checkpoint file.

        Keyword Args
        ------------
        backend_config : dict
            Dictionary containing the backend configuration. Must include the
            fields `'backend'`, `'compilation'` and `'shots'`.

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

    def get_diagram_output(self, diagrams: list[Diagram]) -> np.ndarray:
        """Return the prediction for each diagram using t|ket>.

        Parameters
        ----------
        diagrams : list of :py:class:`Diagram`
            List of lambeq diagrams.

        Raises
        ------
        ValueError
            If `model.weights` or `model.symbols` are not initialised.

        Returns
        -------
        np.ndarray
            Resulting array.

        """
        if len(self.weights) == 0 or not self.symbols:
            raise ValueError('Weights and/or symbols not initialised. '
                             'Instantiate through '
                             '`TketModel.initialise_symbols()` first, '
                             'then call `initialise_weights()`, or load '
                             'from pre-trained checkpoint.')

        lambdified_diagrams = [self._make_lambda(d) for d in diagrams]
        tensors = Circuit.eval(
            *[diag_f(*self.weights) for diag_f in lambdified_diagrams],
            **self.backend_config,
            seed=self._randint()
        )
        self.backend_config['backend'].empty_cache()
        # discopy evals a single diagram into a single result
        # and not a list of results
        if len(diagrams) == 1:
            result = self._normalise(tensors.array)
            return result.reshape(1, *result.shape)
        return np.array([self._normalise(t.array) for t in tensors])

    def forward(self, x: list[Diagram]) -> np.ndarray:
        """Perform default forward pass of a lambeq quantum model.

        In case of a different datapoint (e.g. list of tuple) or additional
        computational steps, please override this method.

        Parameters
        ----------
        x : list of :py:class:`Diagram`
            List of input diagrams.

        Returns
        -------
        np.ndarray
            Array containing model's prediction.

        """
        return self.get_diagram_output(x)
