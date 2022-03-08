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

import os
import pickle
import random
import socket
from abc import ABC, abstractmethod
from datetime import datetime
from math import ceil
from typing import TYPE_CHECKING, Any, Callable, Mapping, Optional, Union

from tqdm import tqdm

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter

from lambeq.training.dataset import Dataset
from lambeq.training.model import Model


def _import_tensorboard_writer() -> None:
    global SummaryWriter
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:  # pragma: no cover
        raise ImportError('tensorboard not found. Please install it using '
                          '`pip install tensorboard`.')


class Trainer(ABC):
    """Base class for a lambeq trainer."""

    def __init__(self,
                 model: Model,
                 loss_function: Callable,
                 epochs: int,
                 evaluate_functions: Optional[Mapping[str, Callable]] = None,
                 evaluate_on_train: bool = True,
                 use_tensorboard: bool = False,
                 log_dir: Optional[Union[str, os.PathLike]] = None,
                 from_checkpoint: bool = False,
                 verbose: str = 'text',
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
        evaluate_on_train : bool, default: True
            Evaluate the metrics on the train dataset.
        use_tensorboard : bool, default: False
            Use Tensorboard for visualisation of the training logs.
        log_dir : str or PathLike, optional
            Location of model checkpoints (and tensorboard log). Default is
            `runs/**CURRENT_DATETIME_HOSTNAME**`.
        from_checkpoint : bool, default: False
            Starts training from the checkpoint, saved in the log_dir.
        verbose : str, default: \'text\',
            Controls the form of progress tracking for the trainer. Set to
            \'text\` for text outputs, \'progress\' for a progress bar, or
            \'suppress\' to have no output.
        seed : int, optional
            Random seed.

        """
        if log_dir is None:
            current_time = datetime.now().strftime('%b%d_%H-%M-%S')
            log_dir = os.path.join(
                'runs', current_time + '_' + socket.gethostname())
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        self.model = model
        self.loss_function = loss_function
        self.epochs = epochs
        self.evaluate_functions = evaluate_functions
        self.evaluate_on_train = evaluate_on_train
        self.use_tensorboard = use_tensorboard
        self.from_checkpoint = from_checkpoint
        self.verbose = verbose
        self.seed = seed

        self.train_costs: list[float] = []
        self.train_epoch_costs: list[float] = []
        self.train_results: dict[str, list[Any]] = {}
        self._train_results_epoch: dict[str, list[Any]] = {}

        self.val_costs: list[float] = []
        self.val_results: dict[str, list[Any]] = {}
        self._val_results_epoch: dict[str, list[Any]] = {}

        if self.evaluate_functions is not None:
            for name in self.evaluate_functions:
                self.val_results[name] = []
                self._val_results_epoch[name] = []
                self.train_results[name] = []
                self._train_results_epoch[name] = []

        verbose_fields = ('progress', 'suppress', 'text')
        if self.verbose not in verbose_fields:
            raise ValueError('The `verbose flag must contain any of the '
                             f'following: {verbose_fields}. `{self.verbose}` '
                             f'was given.')

        if self.verbose == 'text':
            self.printer = print
        else:
            self.printer = lambda *args, **kwargs: None

        if self.seed is not None:
            random.seed(self.seed)

        if self.use_tensorboard:
            _import_tensorboard_writer()
            self.writer = SummaryWriter(log_dir=self.log_dir)

        # load checkpoint
        self.start_epoch = 0
        self.start_step = 0
        if self.from_checkpoint:
            self.checkpoint = self.load_training_checkpoint(self.log_dir)
        else:
            self.model.initialise_weights()

    def load_training_checkpoint(
        self, log_dir: Union[str, os.PathLike]
    ) -> Mapping[str, Any]:
        """Load model from a checkpoint.

        Parameters
        ----------
        log_dir : str or PathLike
            The path to the `model.lt` checkpoint file.

        Returns
        -------
        mapping of str to any
            The checkpoint information.

        Raises
        ------
        FileNotFoundError
            If the file does not exists.
        """
        self.printer("Restore last checkpoint...")
        checkpoint_path = os.path.join(log_dir, 'model.lt')
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, 'rb') as ckp:
                checkpoint = pickle.load(ckp)
            self.model.weights = checkpoint['model_weights']
            self.model.symbols = checkpoint['model_symbols']
            self.train_costs = checkpoint['train_costs']
            self.train_epoch_costs = checkpoint['train_epoch_costs']
            self.train_results = checkpoint['train_results']
            self.val_costs = checkpoint['val_costs']
            self.val_results = checkpoint['val_results']
            self.start_epoch = checkpoint['epoch']
            self.start_step = checkpoint['step']
            if self.seed is not None:
                random.setstate(checkpoint['random_state'])
            self.printer("Checkpoint restored successfully!")
            return checkpoint
        else:
            raise FileNotFoundError('Checkpoint not found! Check path '
                                    f'{checkpoint_path}')

    def save_checkpoint(self,
                        save_dict: Mapping[str, Any],
                        log_dir: Union[str, os.PathLike]) -> None:
        """Save checkpoint.

        Parameters
        ----------
        save_dict : mapping of str to any
            Mapping containing the checkpoint information.
        log_dir : str or PathLike
            The path where to store the `model.lt` checkpoint file.

        """
        add_info = self._add_extra_chkpoint_info()
        appendix = add_info if add_info is not None else {}
        with open(os.path.join(log_dir, 'model.lt'), 'wb') as ckp:
            pickle.dump({**save_dict, **appendix}, ckp)

    @abstractmethod
    def _add_extra_chkpoint_info(self) -> Mapping[str, Any]:
        """Add any additional information to the training checkpoint. These
        might include model-specific information like the random state of the
        backend or the state of the optimiser.

        Returns
        -------
        mapping of str to any
            Mapping containing the extra information to save.

        """

    @abstractmethod
    def _load_extra_chkpoint_info(self,
                                  checkpoint: Mapping[str, Any]) -> None:
        """Load the additional checkpoint information that was previously
        added by calling the method `_add_checkpoint_info()`.

        Parameters
        ----------
        checkpoint : mapping of str to any
            Mapping containing the checkpoint information.

        """

    @abstractmethod
    def training_step(self,
                      batch: tuple[list[Any], Any]) -> tuple[Any, float]:
        """Performs a training step.

        Parameters
        ----------
        batch : tuple of list and any
            Current batch.

        Returns
        -------
        Tuple of any and float
            The model predictions and the calculated loss.

        """

    @abstractmethod
    def validation_step(
            self, batch: tuple[list[Any], Any]) -> tuple[Any, float]:
        """Performs a validation step.

        Parameters
        ----------
        batch : tuple of list and any
            Current batch.

        Returns
        -------
        Tuple of any and float
            The model predictions and the calculated loss.

        """

    def fit(self,
            train_dataset: Dataset,
            val_dataset: Optional[Dataset] = None,
            evaluation_step: int = 1) -> None:
        """Fit the model on the training data and, optionally,
        evaluates it on the validation data.

        Parameters
        ----------
        train_dataset : :py:class:`Diagram`
            Dataset used for training.
        val_dataset : :py:class:`Diagram`, optional
            Validation dataset.
        evaluation_step : int, default: 1
            Sets the intervals at which the metrics are evaluated on the
            validation dataset.

        """
        if self.from_checkpoint:
            self._load_extra_chkpoint_info(self.checkpoint)

        def writer_helper(*args: Any) -> None:
            if self.use_tensorboard:
                self.writer.add_scalar(*args)

        # initialise progress bar
        step = self.start_step
        batches_per_epoch = ceil(len(train_dataset)/train_dataset.batch_size)
        pbar = tqdm(total=batches_per_epoch * self.epochs,
                    desc='Epoch -, batch -, loss: -',
                    disable=self.verbose != 'progress')
        pbar.update(self.start_epoch * batches_per_epoch)

        # start training loop
        for epoch in range(self.start_epoch, self.epochs):
            epoch_loss: float = 0.0
            for batch_idx, batch in enumerate(train_dataset):
                step += 1
                x, y_label = batch
                y_hat, loss = self.training_step(batch)
                if (self.evaluate_on_train and
                        self.evaluate_functions is not None):
                    for metr, func in self.evaluate_functions.items():
                        res = func(y_hat, y_label)
                        self._train_results_epoch[metr].append(len(x)*res)
                self.printer(f'Epoch: {epoch+1}, Batch: {batch_idx+1}, '
                             f'train/loss: {loss:.4f}')
                epoch_loss += len(batch[0])*loss
                writer_helper('train/step_loss', loss, step)
                pbar.set_description(f'Epoch: {epoch+1}, '
                                     f'Batch: {batch_idx+1}, '
                                     f'loss: {loss:.4f}')
                pbar.update(1)
            self.train_epoch_costs.append(epoch_loss/len(train_dataset))
            writer_helper('train/epoch_loss',
                          self.train_epoch_costs[-1], epoch+1)

            # evaluate on train
            if (self.evaluate_on_train and
                    self.evaluate_functions is not None):
                for name in self._train_results_epoch:
                    self.train_results[name].append(
                        sum(self._train_results_epoch[name])/len(train_dataset)
                    )
                    self._train_results_epoch[name] = []  # reset
                    writer_helper(
                        f'train/{name}', self.train_results[name][-1],
                        epoch+1)
                    self.printer(
                        f'Epoch: {epoch+1}, train/{name}: '
                        f'{self.train_results[name][-1]:.4f}')

            # evaluate metrics on validation data
            if val_dataset is not None:
                if epoch % evaluation_step == 0:
                    val_loss: float = 0.0
                    for v_batch in val_dataset:
                        x_val, y_label_val = v_batch
                        y_hat_val, cur_loss = self.validation_step(v_batch)
                        val_loss += cur_loss * len(x_val)
                        if self.evaluate_functions is not None:
                            for metr, func in self.evaluate_functions.items():
                                res = func(y_hat_val, y_label_val)
                                self._val_results_epoch[metr].append(
                                    len(x_val)*res)
                    val_loss /= len(val_dataset)
                    self.val_costs.append(val_loss)
                    self.printer(f'Epoch: {epoch+1}, val/loss: {val_loss:.4f}')
                    writer_helper('val/loss', val_loss, epoch+1)

                    if self.evaluate_functions is not None:
                        for name in self._val_results_epoch:
                            self.val_results[name].append(
                                sum(self._val_results_epoch[name])
                                    /len(val_dataset)
                            )
                            self._val_results_epoch[name] = []  # reset
                            writer_helper(
                                f'val/{name}', self.val_results[name][-1],
                                epoch+1)
                            self.printer(
                                f'Epoch: {epoch+1}, val/{name}: '
                                f'{self.val_results[name][-1]:.4f}')

            # save checkpoint info
            save_dict = {'epoch': epoch+1,
                         'model_weights': self.model.weights,
                         'model_symbols': self.model.symbols,
                         'train_costs': self.train_costs,
                         'train_epoch_costs': self.train_epoch_costs,
                         'train_results': self.train_results,
                         'val_costs': self.val_costs,
                         'val_results': self.val_results,
                         'random_state': random.getstate(),
                         'step': step}
            self.printer("Storing checkpoint...")
            self.save_checkpoint(save_dict, self.log_dir)
            self.printer("Storing checkpoint finished!")
        pbar.close()
        self.printer("\nTraining successful!")
