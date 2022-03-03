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
PytorchModel
============
Module implementing a basic lambeq model based on a Pytorch backend.

"""
from __future__ import annotations

from copy import deepcopy
from typing import Optional

import tensornetwork as tn
import torch
from discopy import Tensor
from discopy.tensor import Diagram
from torch import nn

from lambeq.ansatz.base import Symbol
from lambeq.training.model import Model


class PytorchModel(Model, nn.Module):
    """A lambeq model for the classical pipeline using the Pytorch backend."""

    def __init__(self, diagrams: list[Diagram],
                 seed: Optional[int] = None) -> None:
        """Initialise a ClassicalModel.

        Parameters
        ----------
        diagrams : list of :py:class:`Diagram`
            List of lambeq diagrams.
        seed : int, optional
            Random seed.
        """
        Model.__init__(self, diagrams, seed)
        nn.Module.__init__(self)

        if self.seed is not None:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)

        Tensor.np = torch
        tn.set_default_backend('pytorch')
        torch.set_default_tensor_type(torch.DoubleTensor)
        torch.array = torch.as_tensor   # type: ignore

        self.weights = nn.ParameterList(
            [nn.Parameter(torch.rand(w.size, requires_grad=True))
                for w in self.symbols])

        self.lambdas = {
            circ: circ.lambdify(*self.symbols) for circ in self.diagrams}

    def get_diagram_output(self, diagrams: list[Diagram]) -> torch.Tensor:
        """Perform the tensor contraction of each diagram using tensornetwork.

        Parameters
        ----------
        diagrams : list of :py:class:`Diagram`
            List of lambeq diagrams.

        Returns
        -------
        torch.Tensor
            Resulting tensor.
                for w in self.symbols])

        """
        parameters = {k: v for k, v in zip(self.symbols, self.weights)}
        diagrams = deepcopy(diagrams)
        for diagram in diagrams:
            for b in diagram._boxes:
                if isinstance(b._data, Symbol):
                    try:
                        b._data = parameters[b._data]
                        b._free_symbols = {}
                    except:
                        raise KeyError(f'Unknown symbol {b._data!r}.')

        return torch.stack(
            [tn.contractors.auto(*d.to_tn()).tensor
                for d in diagrams])

    def forward(self, x: list[Diagram]) -> torch.Tensor:
        """Perform default forward pass of a lambeq model by contracting tensors.

        In case of a different datapoint (e.g. list of tuple) or additional
        computational steps, please override this method.

        Parameters
        ----------
        x : list of :py:class:`Diagram`
            List of input diagrams.

        Returns
        -------
        torch.Tensor
            Tensor containing model's prediction.

        """
        return self.get_diagram_output(x)
