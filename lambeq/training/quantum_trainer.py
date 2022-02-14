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
QuantumTrainer
==============
A trainer that wraps the training loop of a :py:class:`QuantumModel` or
a :py:class:`ECSQuantumModel`.

"""
from __future__ import annotations

import os
import pickle
import socket
from datetime import datetime
from typing import (Any, Callable, Mapping, Optional, Type, TYPE_CHECKING,
                    Union)

import numpy
if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter

from lambeq.training.dataset import Dataset
from lambeq.training.quantum_model import QuantumModel
from lambeq.training.ecs_quantum_model import ECSQuantumModel
from lambeq.training.trainer import Trainer
from lambeq.training.optimiser import Optimiser


def _import_tensorboard_writer() -> None:
    global SummaryWriter
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:  # pragma: no cover
        raise ImportError('tensorboard not found. Please install it using '
                          '`pip install tensorboard`.')


class QuantumTrainer(Trainer):
    """A Trainer for the quantum pipeline."""

    model: Union[QuantumModel, ECSQuantumModel]

    def __init__(
            self,
            model: Union[QuantumModel, ECSQuantumModel],
            loss_function: Callable,
            epochs: int,
            optimizer: Type[Optimiser],
            optim_hyperparams: dict[str, float],
            evaluate_functions: Optional[Mapping[str, Callable]] = None,
            use_tensorboard: bool = False,
            log_dir: Optional[str] = None,
            verbose: bool = False,
            seed: Optional[int] = None) -> None:
        """Initialise a :py:class:`.Trainer` instance using a quantum backend.

        Parameters
        ----------
        model : Model
            A lambeq Model.
        loss_function : callable
            A loss function.
        epochs : int
            Number of training epochs
        optimizer : Optimiser
            A optimizer of type `lambeq.training.Optimiser`.
        evaluate_functions : mapping of str to callable, optional
            Mapping of evaluation metric functions from their names.
            Structure [{\"metric\": func}].
            Each function takes the prediction \"y_hat\" and the label \"y\" as
            input.
            The validation step calls \"func(y_hat, y)\".
        use_tensorboard : bool, default: False
            Use Tensorboard for visualisation of the training logs.
        log_dir : str, optional
            Location of model checkpoints (and tensorboard log). Default is
            `runs/**CURRENT_DATETIME_HOSTNAME**`.
        verbose : bool, default: True,
            Setting verbose to False surpresses the commandline output and
            prints dots as status bar.
        seed : int, optional
            Random seed.

        """
        super().__init__(model,
                         loss_function,
                         epochs,
                         evaluate_functions,
                         verbose,
                         seed)
        if log_dir is None:
            current_time = datetime.now().strftime('%b%d_%H-%M-%S')
            log_dir = os.path.join(
                'runs', current_time + '_' + socket.gethostname())

        self.use_tensorboard = use_tensorboard
        self.log_dir = log_dir
        self.verbose = verbose
        self.optimizer = optimizer(self.model,
                                   optim_hyperparams,
                                   self.loss_function,
                                   seed=self.seed)

        os.makedirs(self.log_dir, exist_ok=True)
        if self.use_tensorboard:
            _import_tensorboard_writer()
            self.writer = SummaryWriter(log_dir=self.log_dir)

    def validation_step(self,
                        batch: tuple[list[Any], list[numpy.ndarray]]) -> float:
        """Performs a validation step.

        Parameters
        ----------
        batch : tuple of list and list of numpy.ndarray
            Current batch.

        Returns
        -------
        float
            Calculated loss.

        """
        x, y = batch
        y_hat = self.model.forward(x)
        loss = self.loss_function(y_hat, y)

        if self.evaluate_functions is not None:
            for metric, func in self.evaluate_functions.items():
                res = func(y_hat, y)
                self.val_results_current[metric].append(res * len(y))
        return loss

    def training_step(self,
                      batch: tuple[list[Any], list[numpy.ndarray]]) -> float:
        """Performs a training step.

        Parameters
        ----------
        batch : tuple of list and list of numpy.array
            Current batch.

        Returns
        -------
        float
            Calculated loss.

        """
        loss = self.optimizer.backward(batch)
        self.train_costs.append(loss)
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss

    def fit(self,
            train_dataset: Dataset,
            val_dataset: Optional[Dataset] = None) -> None:
        """Fit the model on the training data and, optionally,
        evaluates it on the validation data.

        Parameters
        ----------
        train_dataset : :py:class:`Diagram`
            Dataset used for training.
        val_dataset : :py:class:`Diagram`, optional
            Validation dataset.

        """
        def writer_helper(*args: Any) -> None:
            if self.use_tensorboard:
                self.writer.add_scalar(*args)
            else:
                pass

        step = 0
        for epoch in range(self.epochs):
            epoch_loss: float = 0.0
            for batch_idx, batch in enumerate(train_dataset):
                step += 1
                loss = self.training_step(batch)
                self.printer(f'Epoch: {epoch+1}, Batch: {batch_idx+1}, '
                             f'train/loss: {loss:.4f}')
                epoch_loss += len(batch[0])*loss
                writer_helper('train/step_loss', loss, step)
            self.train_epoch_costs.append(epoch_loss/len(train_dataset))
            writer_helper('train/epoch_loss',
                          self.train_epoch_costs[-1], epoch+1)

            # save model
            with open(self.log_dir + '/model.pkl', 'wb') as fb:
                pickle.dump({'symbols': self.model.symbols,
                             'weights': self.model.weights}, fb)

            if val_dataset is not None:
                val_loss: float = 0.0
                for v_batch in val_dataset:
                    val_loss += self.validation_step(v_batch) * len(v_batch[0])
                val_loss /= len(val_dataset)
                self.val_costs.append(loss)
                self.printer(f'Epoch: {epoch+1}, val/loss: {val_loss:.4f}')
                writer_helper('val/loss', val_loss, epoch+1)

                if self.evaluate_functions is not None:
                    for name in self.val_results_current:
                        self.val_results[name].append(
                            sum(self.val_results_current[name])/len(val_dataset)
                        )
                        self.val_results_current[name] = []  # reset
                        writer_helper(
                            f'val/{name}', self.val_results[name][-1],
                            epoch+1)
                        self.printer(
                            f'Epoch: {epoch+1}, val/{name}: '
                            f'{self.val_results[name][-1]:.4f}')
        print("\nTraining successful!")
