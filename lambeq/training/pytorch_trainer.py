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
PytorchTrainer
==============
A trainer that wraps the training loop of a :py:class:`PytorchModel`.

"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime
import os
import socket
from typing import Any, Optional, TYPE_CHECKING

import torch
if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter

from lambeq.training.dataset import Dataset
from lambeq.training.pytorch_model import PytorchModel
from lambeq.training.trainer import Trainer


def _import_tensorboard_writer() -> None:
    global SummaryWriter
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:  # pragma: no cover
        raise ImportError('tensorboard not found. Please install it using '
                          '`pip install tensorboard`.')


class PytorchTrainer(Trainer):
    """A PyTorch trainer for the classical pipeline."""

    model: PytorchModel

    def __init__(
            self,
            model: PytorchModel,
            loss_function: Callable,
            epochs: int,
            optimizer: type[torch.optim.Optimizer] = torch.optim.AdamW,
            learning_rate: float = 1e-3,
            device: int = -1,
            evaluate_functions: Optional[Mapping[str, Callable]] = None,
            use_tensorboard: bool = False,
            log_dir: Optional[str] = None,
            seed: Optional[int] = None) -> None:
        """Initialise a trainer instance using the PyTorch backend.

        Parameters
        ----------
        model : PytorchModel
            A lambeq Model using the PyTorch backend for tensor computation.
        loss_function : callable
            A PyTorch loss function from `torch.nn`.
        optimizer : torch.optim.Optimizer, default: torch.optim.AdamW
            A PyTorch optimizer from `torch.optim`.
        learning_rate : float, default: 1e-3
            The learning rate for training.
        epochs : int
            Number of training epochs.
        device : int, default: -1
            CUDA device ID used for tensor operation speed-up. A negative value
            uses the CPU.
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
        seed : int, optional
            Random seed.

        """
        super().__init__(model,
                         loss_function,
                         epochs,
                         evaluate_functions,
                         seed)
        if log_dir is None:
            current_time = datetime.now().strftime('%b%d_%H-%M-%S')
            log_dir = os.path.join(
                'runs', current_time + '_' + socket.gethostname())

        self.learning_rate = learning_rate
        self.device = torch.device('cpu' if device < 0 else f'cuda:{device}')
        self.use_tensorboard = use_tensorboard
        self.log_dir = log_dir
        self.optimizer = optimizer(self.model.parameters(),  # type: ignore
                                   lr=self.learning_rate)   # type: ignore
        self.model.to(self.device)

        os.makedirs(self.log_dir, exist_ok=True)
        if self.use_tensorboard:
            _import_tensorboard_writer()
            self.writer = SummaryWriter(log_dir=self.log_dir)

    def validation_step(self,
                        batch: tuple[list[Any], list[torch.Tensor]]) -> float:
        """Performs a validation step.

        Parameters
        ----------
        batch : tuple of list and list of torch.Tensor
            Current batch.

        Returns
        -------
        float
            Calculated loss.

        """
        x, y = batch
        if not isinstance(y, list):
            raise TypeError(
                f'Targets must be of type `list[Tensor]` not `{type(y)}`')

        with torch.no_grad():
            y_hat = self.model(x)
            loss = self.loss_function(y_hat, torch.stack(y))
            self.val_costs.append(loss.item())

        if self.evaluate_functions is not None:
            for metric, func in self.evaluate_functions.items():
                res = func(y_hat, y)
                self.val_results[metric].append(res)
        return loss.item()

    def training_step(self,
                      batch: tuple[list[Any], list[torch.Tensor]]) -> float:
        """Performs a training step.

        Parameters
        ----------
        batch : tuple of list and list of torch.Tensor
            Current batch.

        Returns
        -------
        float
            Calculated loss.

        """
        x, y = batch
        if not isinstance(y, list):
            raise TypeError(
                f'Targets must be of type `list[Tensor]` not `{type(y)}`')

        y_hat = self.model(x)
        loss = self.loss_function(y_hat, torch.stack(y))
        self.train_costs.append(loss.item())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

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
            epoch_loss = 0.0
            for batch_idx, batch in enumerate(train_dataset):
                step += 1
                loss = self.training_step(batch)
                print(f'Epoch: {epoch+1}, Batch: {batch_idx+1}, '
                      f'train/loss: {loss:.4f}')
                epoch_loss += len(batch[0])*loss
                writer_helper('train/step_loss', loss, step)
            self.train_epoch_costs.append(epoch_loss/len(train_dataset))
            writer_helper('train/epoch_loss',
                          self.train_epoch_costs[-1], epoch+1)

            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': self.train_costs[-1],
                }, self.log_dir + '/model.pt')

            if val_dataset is not None:
                val_loss = self.validation_step(val_dataset[:])
                print(f'Epoch: {epoch+1}, val/loss: {val_loss:.4f}')
                writer_helper('val/loss', val_loss, epoch+1)

                if self.evaluate_functions is not None:
                    for name in self.val_results:
                        writer_helper(
                            f'val/{name}', self.val_results[name][-1],
                            epoch+1)
                        print(
                            f'Epoch: {epoch+1}, val/{name}: '
                            f'{self.val_results[name][-1]:.4f}')
