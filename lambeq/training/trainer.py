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
Trainer
=======

Module that contains the base class for a lambeq trainer.

Subclass :py:class:`Lambeq` to define a custom trainer.

"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from typing import Any, Optional

from lambeq.training.dataset import Dataset
from lambeq.training.model import Model


class Trainer(ABC):
    """Base class for a lambeq trainer."""

    def __init__(self,
                 model: Model,
                 loss_function: Callable,
                 epochs: int,
                 evaluate_functions: Optional[Mapping[str, Callable]] = None,
                 seed: Optional[int] = None) -> None:
        """Initialise a lambeq trainer.

        Parameters
        ----------
        model : Model
            A lambeq Model.
        loss_function : callable
            A loss function to compare the prediction to the true label.
        epochs : int
            Number of training epochs.
        evaluate_functions : mapping of str to callable, optional
            Mapping of evaluation metric functions from their names.
        seed : int, optional
            Random seed.

        """
        self.model = model
        self.loss_function = loss_function
        self.epochs = epochs
        self.evaluate_functions = evaluate_functions
        self.seed = seed

        self.train_costs: list[float] = []
        self.train_epoch_costs: list[float] = []

        self.val_costs: list[float] = []
        self.val_results: dict[str, list[Any]] = {}

        if self.evaluate_functions is not None:
            for name in self.evaluate_functions:
                self.val_results[name] = []

    @abstractmethod
    def fit(self,
            train_dataset: Dataset,
            val_dataset: Optional[Dataset] = None) -> None:
        """Implement the training routine of a LambeqTrainer subclass.

        Parameters
        ----------
        train_dataset : Dataset
            Dataset used for training.
        val_dataset : Dataset, optional
            Dataset used for validation.

        """
