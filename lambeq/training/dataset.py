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
Dataset
=======
A module containing a Dataset class for training lambeq models.

"""

from math import ceil
import random
from typing import Any, Iterator, List, Optional, Tuple, Union


class Dataset:
    """Dataset class for the training of a lambeq model."""

    def __init__(self,
                 data: List[Any],
                 targets: List[Any],
                 batch_size: int = 0,
                 shuffle: bool = True,
                 rnd_seed: Optional[int] = None) -> None:
        """Initialise a Dataset for lambeq training.

        Parameters
        ----------
        data : list
            Data used for training.
        targets : list
            List of labels.
        batch_size : int, default: 0
            Batch size for batch generation, by default full dataset.
        shuffle : bool, default: True
            Enable data shuffling during training.
        rnd_seed : int, optional
            Random seed.

        """
        self.data = data
        self.targets = targets
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rnd_seed = rnd_seed

        if self.batch_size == 0:
            self.batch_size = len(self.data)

        if self.rnd_seed is not None:
            random.seed(self.rnd_seed)

        self.batches_per_epoch = ceil(len(self.data) / self.batch_size)

    def __getitem__(self, index: Union[int, slice]) -> Tuple[Any, Any]:
        """Get a single item or a subset from the dataset."""
        x = self.data[index]
        y = self.targets[index]
        return x, y

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self) -> Iterator[Tuple[List[Any], List[Any]]]:
        """Iterate over data batches.

        Yields
        ------
        Tuple of list and list
            An iterator that yields data batches (X_batch, y_batch).

        """

        new_data, new_targets = self.data, self.targets

        if self.shuffle:
            new_data, new_targets = self.shuffle_data(new_data, new_targets)

        for start_idx in range(0, len(self.data), self.batch_size):
            yield (new_data[start_idx : start_idx+self.batch_size],
                   new_targets[start_idx : start_idx+self.batch_size])

    @staticmethod
    def shuffle_data(data: List[Any],
                     targets: List[Any]) -> Tuple[List[Any], List[Any]]:
        """Shuffle a given dataset.

        Parameters
        ----------
        data : list
            List of data points.
        targets : list
            List of labels.

        Returns
        -------
        Tuple of list and list
            The shuffled dataset.

        """
        joint_list = list(zip(data, targets))
        random.shuffle(joint_list)
        data, targets = zip(*joint_list)
        return list(data), list(targets)
