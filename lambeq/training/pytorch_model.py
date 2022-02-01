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

from typing import List, Optional

import tensornetwork as tn
import torch
from discopy import Tensor
from discopy.tensor import Diagram
from torch import nn

from lambeq.training.model import Model


class PytorchModel(Model, nn.Module):
    """A lambeq model for the classical pipeline using the Pytorch backend."""

    def __init__(self, seed: Optional[int] = None) -> None:
        """Initialise an instance of the :py:class:`PytorchModel` class.

        Parameters
        ----------
        seed : int, optional
            Random seed.
        """
        Model.__init__(self, seed)
        nn.Module.__init__(self)

        if self.seed is not None:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)

        Tensor.np = torch
        tn.set_default_backend('pytorch')
        torch.set_default_tensor_type(torch.DoubleTensor)
        torch.array = torch.as_tensor   # type: ignore

    def tensorise(self) -> None:
        """Initialize tensors for each word in the vocabulary."""
        if not self.vocab:
            raise ValueError('Vocabulary not initialized. Call method '
                             f'.prepare_vocab() of "{self._get_name()}" '
                             'instance first.')

        self.word_params = nn.ParameterList(
            [nn.Parameter(torch.rand(w.size, requires_grad=True))
                for w in self.vocab])

    @staticmethod
    def contract(diagrams: List[Diagram]) -> torch.Tensor:
        """Perform the tensor contraction of each diagram.

        Parameters
        ----------
        diagrams : list of :py:class:`Diagram`
            List of lambeq diagrams.

        Returns
        -------
        torch.Tensor
            Resulting tensor.

        """
        return torch.stack(
            [d.eval(contractor=tn.contractors.auto).array for d in diagrams])

    def forward(self, x: List[Diagram]) -> torch.Tensor:
        """Perform default forward pass of a lambeq model.

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
        tensor_nets = self.lambdify(x)
        return self.contract(tensor_nets)
