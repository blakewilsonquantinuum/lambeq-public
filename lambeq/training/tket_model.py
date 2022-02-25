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
TketModel
=========
Module implementing a lambeq model based on a quantum backend, via `tket`.

"""
from __future__ import annotations

from typing import Optional

import numpy as np
from discopy.quantum import Circuit, Id, Measure
from discopy.tensor import Diagram, Tensor

from lambeq.training.model import Model


class TketModel(Model):
    """A lambeq model for either shot-based simulations of a quantum
    pipeline or experiments run on quantum hardware using `tket`."""

    def __init__(self, diagrams: list[Diagram], backend_config,
                 seed: Optional[int] = None) -> None:
        """Initialise TketModel based on the `t|ket>` backend.

        Parameters
        ----------
        diagrams : list of :py:class:`Diagram`
            List of lambeq diagrams.
        seed : int, optional
            Random seed.

        """
        super().__init__(diagrams, seed)

        self.rng = np.random.default_rng(seed)
        Tensor.np = np

        fields = ('backend', 'compilation', 'shots')
        if any(field not in backend_config for field in fields):
            raise ValueError('Missing arguments in backend configuation.')
        self.backend_config = backend_config

        assert all(w.size == 1 for w in self.symbols)
        self.weights = np.array(self.rng.random(len(self.symbols)))

        measured_diagrams = [d >> Id().tensor(*[Measure()] * len(d.cod))
                             for d in self.diagrams]
        self.lambdas = {
            dig: circ.lambdify(*self.symbols)
            for dig, circ in zip(self.diagrams, measured_diagrams)}

    def _randint(self, rng, low=-1 << 63, high=(1 << 63)-1):
        return rng.integers(low, high)

    def _normalise(self, predictions: np.ndarray) -> np.ndarray:
        """apply smoothing to predictions"""
        predictions = np.abs(predictions) + 1e-9
        return predictions / predictions.sum()

    def get_diagram_output(self, circuits: list[Diagram]) -> np.ndarray:
        """Return the prediction for each diagram using t|ket>.

        Parameters
        ----------
        circuits : list of :py:class:`Diagram`
            List of lambeq circuits.

        Returns
        -------
        np.ndarray
            Resulting array.

        """
        tensors = Circuit.eval(
            *[self.lambdas[d](*self.weights) for d in circuits],
            **self.backend_config,
            seed=self._randint(self.rng)
        )
        self.backend_config['backend'].empty_cache()
        # discopy evals a single diagram into a single result
        # and not a list of results
        if len(circuits) == 1:
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
            List of input circuits.

        Returns
        -------
        np.ndarray
            Array containing model's prediction.

        """
        return self.get_diagram_output(x)
