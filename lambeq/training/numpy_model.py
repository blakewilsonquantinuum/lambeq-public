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

from typing import Any, Callable, Optional

import numpy
from discopy import Tensor
from discopy.tensor import Diagram

from lambeq.training.model import Model


class NumpyModel(Model):
    """A lambeq model for an exact classical simulation of a
    quantum pipeline."""

    def __init__(self, diagrams: list[Diagram],
                 seed: Optional[int] = None) -> None:
        """Initialise an NumpyModel. If you want to use jax support,
        use

        ```
        import jax
        from discopy import Tensor
        Tensor.np = jax.numpy
        ```

        Parameters
        ----------
        diagrams : list of :py:class:`Diagram`
            List of lambeq circuits.
        seed : int, optional
            Random seed.

        """
        super().__init__(diagrams, seed)
        self.np = Tensor.np
        self.rng = numpy.random.default_rng(seed)

        assert all(w.size == 1 for w in self.symbols)
        self.weights = self.np.array(self.rng.random(len(self.symbols)))

        self.lambdas = {circ: self._make_lambda(circ) for circ in self.diagrams}
        if Tensor.np.__name__ == 'jax.numpy':
            from jax import jit
            self.lambdas = {circ: jit(f) for circ, f in self.lambdas.items()}

    def _normalise(self, predictions: numpy.ndarray) -> numpy.ndarray:
        """Apply smoothing to predictions."""
        predictions = self.np.abs(predictions) + 1e-9
        return predictions / predictions.sum()

    def _make_lambda(self, circuit: Diagram) -> Callable[[Any], Any]:
        """Make lambda that evaluates the provided circuit."""
        return lambda *x: (
            self._normalise(circuit.lambdify(*self.symbols)(*x).eval().array))

    def get_diagram_output(self, diagrams: list[Diagram]) -> numpy.ndarray:
        """Return the exact prediction for each diagram using DisCoPy.

        Parameters
        ----------
        circuits : list of :py:class:`Diagram`
            List of lambeq circuits.

        Returns
        -------
        np.ndarray
            Resulting array

        """
        return numpy.array([self.lambdas[d](*self.weights) for d in diagrams])

    def forward(self, x: list[Diagram]) -> numpy.ndarray:
        """Perform default forward pass of a lambeq model.

        In case of a different datapoint (e.g. list of tuple) or additional
        computational steps, please override this method.

        Parameters
        ----------
        x : list of :py:class:`Diagram`
            List of input circuits.

        Returns
        -------
        numpy.ndarray
            Array containing model's prediction.

        """
        return self.get_diagram_output(x)
