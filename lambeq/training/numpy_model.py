# Copyright 2022 Cambridge Quantum Computing Ltd.
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
NumpyModel
==========
Module implementing a lambeq model for an exact classical simulation of
a quantum pipeline.

In contrast to the shot-based :py:class:`TketModel`, the state vectors are
calculated classically and stored such that the complex vectors defining the
quantum states are accessible. The results of the calculations are exact i.e.
noiseless and not shot-based.

"""
from __future__ import annotations

import os
import pickle
from typing import TYPE_CHECKING, Any, Callable, Mapping, Union

import numpy
from discopy import Tensor
from discopy.tensor import Diagram
from sympy import default_sort_key

from lambeq.training.model import Model


class NumpyModel(Model):
    """A lambeq model for an exact classical simulation of a
    quantum pipeline."""

    SMOOTHING = 1e-9

    def __init__(self, **kwargs) -> None:
        """Initialise an NumpyModel. If you want to use jax support,
        use

        ```
        import jax
        from discopy import Tensor
        Tensor.np = jax.numpy
        ```

        """
        super().__init__()
        self.np = Tensor.np
        self.lambdas: Mapping[Diagram, Callable] = {}

    @classmethod
    def initialise_symbols(cls, diagrams: list[Diagram], **kwargs):
        """Extract the symbols from a list of :py:class:`Diagram`s and creates
        a dictionary that maps the diagrams to the according lambda functions.

        Parameters
        ----------
        diagrams : list of :py:class:`Diagram`
            List of lambeq diagrams.

        """
        model = cls(**kwargs)
        model.symbols = sorted(
            {sym for circ in diagrams for sym in circ.free_symbols},
            key=default_sort_key)
        model.lambdas = {d: model._make_lambda(d) for d in diagrams}
        return model

    def _make_lambda(self, diagram: Diagram) -> Callable[[Any], Any]:
        """Make lambda function that evaluates the provided diagram.

        Raises
        ------
        ValueError
            If `model.symbols` are not initialised.

        """
        if not self.symbols:
            raise ValueError('Symbols not initialised. Instantiate through '
                             '`NumpyModel.initialise_symbols()`.')
        diag_f = lambda *x: (
            self._normalise(diagram.lambdify(*self.symbols)(*x).eval().array))
        if Tensor.np.__name__ == 'jax.numpy':
            from jax import jit
            return jit(diag_f)
        return diag_f

    def _normalise(self, predictions: numpy.ndarray) -> numpy.ndarray:
        """Apply smoothing to predictions."""
        predictions = self.np.abs(predictions) + self.SMOOTHING
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
                             '`NumpyModel.initialise_symbols()`.')
        assert all(w.size == 1 for w in self.symbols)
        self.weights = self.np.array(
            numpy.random.uniform(size=len(self.symbols)))

    @classmethod
    def load_from_checkpoint(cls,
                             checkpoint_path: Union[str, os.PathLike],
                             **kwargs) -> NumpyModel:
        """Load the model weights and symbols from a training checkpoint.

        Parameters
        ----------
        checkpoint_path : str or PathLike
            Path that points to the checkpoint file.

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

    def get_diagram_output(self, diagrams: list[Diagram]) -> numpy.ndarray:
        """Return the exact prediction for each diagram.

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
            Resulting array

        """
        if len(self.weights) == 0 or not self.symbols:
            raise ValueError('Weights and/or symbols not initialised. '
                             'Instantiate through '
                             '`NumpyModel.initialise_symbols()` first, '
                             'then call `initialise_weights()`, or load '
                             'from pre-trained checkpoint.')

        lambdified_diagrams = [self.lambdas.get(d, self._make_lambda(d))
                               for d in diagrams]
        return numpy.array([diag_f(*self.weights)
                            for diag_f in lambdified_diagrams])

    def forward(self, x: list[Diagram]) -> numpy.ndarray:
        """Perform default forward pass of a lambeq model.

        In case of a different datapoint (e.g. list of tuple) or additional
        computational steps, please override this method.

        Parameters
        ----------
        x : list of :py:class:`Diagram`
            List of input diagrams.

        Returns
        -------
        numpy.ndarray
            Array containing model's prediction.

        """
        return self.get_diagram_output(x)
