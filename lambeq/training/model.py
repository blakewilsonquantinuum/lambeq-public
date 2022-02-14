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
Model
=====
Module containing the base class for a lambeq model.

"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional
from typing_extensions import Protocol

from discopy.tensor import Diagram
from sympy import default_sort_key


class SizedIterable(Protocol):
    """Custom type for a data that has a length and is iterable."""
    def __len__(self):
        pass

    def __iter__(self):
        pass


class Model(ABC):
    """Model base class.

    Attributes
    ----------
    symbols : list of symbols
        A sorted list of all :py:class:`.Symbol`s occuring in the data.
    weights : SizedIterable
        A data structure containing the numeric values of
        the model's parameters.

    """

    def __init__(self, diagrams: list[Diagram],
                 seed: Optional[int] = None) -> None:
        """Initialise an instance of :py:class:`Model` base class.

        Parameters
        ----------
        diagrams : list of :py:class:`Diagram`
            List of lambeq diagrams.
        seed : int, optional
            Random seed.

        """
        self.diagrams = diagrams
        self.seed = seed
        self.weights: SizedIterable = []

        self.symbols = sorted(
            {sym for circ in diagrams for sym in circ.free_symbols},
            key=default_sort_key)

    @abstractmethod
    def get_diagram_output(self, diagrams: list[Diagram]) -> Any:
        """Return the diagram prediction.

        Parameters
        ----------
        diagrams : list of diagram
            List of lambeq diagrams.

        """

    @abstractmethod
    def forward(self, x: list[Any]) -> Any:
        """Implement default forward pass of model."""
