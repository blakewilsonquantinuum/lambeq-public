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

from abc import ABC, abstractmethod
from typing import Any, Iterable, List, Optional

from discopy.tensor import Diagram
from sympy import default_sort_key

from lambeq.ansatz import Symbol


class Model(ABC):
    """Model base class.

    Attributes
    ----------
    vocab : list of symbols
        A sorted list of all words occuring in the data.
    word_params : Iterable
        A data structure containing the model's parameters.

    """

    def __init__(self, seed: Optional[int] = None) -> None:
        """Initialise an instance of the :py:class:`Model` base class.

        Parameters
        ----------
        seed : int, optional
            Random seed.

        """
        self.vocab: List[Symbol] = []
        self.word_params: Iterable = []
        self.seed = seed

    def prepare_vocab(self, diagrams: List[Diagram]) -> None:
        """Extract the vocabulary from a list of diagrams.

        Parameters
        ----------
        diagrams : list of :py:class:`Diagram`
            List of lambeq diagrams.

        """
        self.vocab = sorted(
            {sym for circ in diagrams for sym in circ.free_symbols},
            key=default_sort_key)

    @abstractmethod
    def tensorise(self) -> None:
        """Initialise the ansatz parameters for each word in the vocabulary.
        It fills the attribute :py:attribute:`self.word_params`.

        """

    def lambdify(self, diagrams: List[Diagram]) -> List[Diagram]:
        """Replace the symbols in a list of diagrams with tensors.

        Parameters
        ----------
        diagrams : list of :py:class:`Diagram`
            List of lambeq diagrams.

        Returns
        -------
        list of :py:class:`Diagram`
            List of lambdified lambeq diagrams.

        """
        if not (self.vocab and self.word_params):
            raise ValueError('Vocabulary/embeddings empty. Call methods '
                             '`.prepare_vocab()` and `.tensorise()` '
                             'of model instance first.')
        return [d.lambdify(*self.vocab)(*self.word_params) for d in diagrams]

    @staticmethod
    @abstractmethod
    def contract(diagrams: List[Diagram]) -> Any:
        """Contract the tensor diagrams.

        Parameters
        ----------
        diagrams : list of diagram
            List of lambeq diagrams.
        """

    @abstractmethod
    def forward(self, x: List[Any]) -> Any:
        """Implement default forward pass of model."""
