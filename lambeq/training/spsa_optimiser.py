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
SPASOptimiser
=============
Module implementing the Simultaneous Perturbation Stochastic Approximation
optimiser.

"""
from __future__ import annotations

from typing import Any, Callable, Optional, Union
from numpy.typing import ArrayLike

import numpy as np
from lambeq.training.numpy_model import NumpyModel
from lambeq.training.optimiser import Optimiser
from lambeq.training.tket_model import TketModel


class SPSAOptimiser(Optimiser):
    """An Optimiser using simultaneous perturbation stochastic approximations.
    See https://ieeexplore.ieee.org/document/705889 for details.
    """

    def __init__(self, model: Union[NumpyModel, TketModel],
                 hyperparams: dict[str, float],
                 loss_fn: Callable[[Any, Any], Any],
                 bounds: Optional[ArrayLike] = None,
                 seed: Optional[int] = None):
        """Initialise the SPSA optimiser.

        The hyperparameters must contain the following key value pairs:

        ```
        hyperparams = {
            'a': A learning rate parameter, float
            'c': The parameter shift scaling factor, float
            'A': A stability constant, approx. 0.01 * Num Training steps, float
        }
        ```

        Parameters
        ----------
        model : NumpyModel or TketModel
            A lambeq model.
        hyperparams : dict of str to float.
            A dictionary containing the models hyperparameters.
        loss_fn : Callable
            A loss function of form `loss(prediction, labels)`
        bounds : ArrayLike, optional
            The range of each of the model\'s parameters.
        seed : int, optional
            Random seed.

        Raises
        ------
        ValueError
            Raises an error if the hyperparameters are not set correctly.
        ValueError
            Raises an error if the length of `bounds` does not match the
            number of the models\'s parameters.

        """
        fields = ('a', 'c', 'A')
        if any(field not in hyperparams for field in fields):
            raise ValueError('Missing arguments in hyperparameter dict'
                             f'configuation. Must contain {fields}.')
        super().__init__(model, hyperparams, loss_fn, bounds, seed)
        self.alpha = 0.602
        self.gamma = 0.101
        self.current_sweep = 1
        self.A = self.hyperparams['A']
        self.a = self.hyperparams['a']
        self.c = self.hyperparams['c']
        self.ak = self.a/(self.current_sweep+self.A)**self.alpha
        self.ck = self.c/(self.current_sweep)**self.gamma

        if self.bounds is None:
            self.project = lambda _: _
        else:
            bds = np.asarray(bounds)
            if len(bds) != len(self.model.weights):
                raise ValueError('Length of `bounds` must be the same as the '
                                 'number of the model\'s parameters')
            self.project = lambda x: np.clip(x, bds[:, 0], bds[:, 1])

        self.rng = np.random.default_rng(self.seed)

    def backward(self, batch: tuple[list[Any], list[np.ndarray]]) -> float:
        """Calculate the gradients of the loss function with respect to the
        model\'s parameters.

        Parameters
        ----------
        batch : tuple of list and list of numpy.ndarray
            Current batch.

        Returns
        -------
        float
            The mean loss of the current SPSA iteration.

        """
        diagrams, targets = batch
        relevant_params = set.union(*[diag.free_symbols for diag in diagrams])
        # the symbolic parameters
        parameters = self.model.symbols
        x = self.model.weights
        # the perturbations
        delta = self.rng.choice([-1, 1], size=len(x))
        mask = [0 if sym in relevant_params else 1 for sym in parameters]
        delta = np.ma.masked_array(delta, mask=mask)
        # calculate gradient
        xplus = self.project(x + self.ck * delta)
        self.model.weights = xplus
        y0 = self.model.forward(diagrams)
        loss0 = self.loss_fn(y0, targets)
        xminus = self.project(x - self.ck * delta)
        self.model.weights = xminus
        y1 = self.model.forward(diagrams)
        loss1 = self.loss_fn(y1, targets)
        if self.bounds is None:
            grad = (loss0 - loss1) / (2*self.ck*delta)
        else:
            grad = (loss0 - loss1) / (xplus-xminus)
        self.gradient += np.ma.filled(grad, fill_value=0)
        # restore parameter value
        self.model.weights = x
        return (loss0+loss1)/2  # return mean loss

    def step(self) -> None:
        """Perform optimisation step."""
        self.model.weights -= self.gradient * self.ak
        self.model.weights = self.project(self.model.weights)
        self.update_hyper_params()
        self.zero_grad()

    def update_hyper_params(self):
        """Update the hyperparameters of the SPSA algorithm."""
        self.current_sweep += 1
        a_decay = (self.current_sweep+self.A)**self.alpha
        c_decay = (self.current_sweep)**self.gamma
        self.ak = self.hyperparams['a']/a_decay
        self.ck = self.hyperparams['c']/c_decay
