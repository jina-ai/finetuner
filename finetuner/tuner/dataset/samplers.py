from collections import defaultdict
from copy import deepcopy
from math import ceil
from random import choices, sample, shuffle
from typing import Sequence, Tuple, Optional

from finetuner.tuner.dataset.base import BaseSampler


class ClassSampler(BaseSampler):
    """A batch sampler that fills the batch with an equal number of items from each
    class.

    It will try to make sure that all the items get used once in a single epoch, and
    that only ``num_items_per_class`` items in a batch come from one class. When there
    would not be enough items left from a single class to fill the batch, items will be
    randomly sampled from other already used items for this class.

    However, some cutoff might occur if there are not enough items left from different
    classes to fill the batch - in this case some items do not get used in that epoch.
    """

    def __init__(
        self,
        labels: Sequence[int],
        batch_size: int,
        num_items_per_class: Optional[int] = None,
    ):
        """Construct the batch sample.

        :param labels: A sequence of items labels, each label should be an integer
            denoting the class of the item
        :param batch_size: How many items to include in a batch
        :param num_items_per_class: How many items per class (unique labels) to include
            in a batch. For example, if ``batch_size`` is 20, and
            ``num_items_per_class`` is 4, the batch will consist of 4 items for each of
            the 5 classes.
        """

        if num_items_per_class is not None and num_items_per_class < 1:
            raise ValueError(
                '`num_items_per_class` must be either None or greater than 0'
            )

        self._num_items_per_class = num_items_per_class or max(
            1, batch_size // len(set(labels))
        )
        self._num_classes = batch_size // self._num_items_per_class

        # Get mapping of labels (classes) and their positions
        self._class_to_labels = defaultdict(list)
        for idx, label in enumerate(labels):
            self._class_to_labels[label].append(idx)

        # Get class groups (ids to use in batches)
        self._cls_group_counts = []
        for key, val in self._class_to_labels.items():
            self._cls_group_counts.append(
                [key, ceil(len(val) / self._num_items_per_class)]
            )

        super().__init__(labels, batch_size)

    def _prepare_batches(self):

        # Shuffle class groups into batches, some cutoff may occur
        counts = deepcopy(self._cls_group_counts)
        group_batches = []
        while len(counts) >= self._num_classes:
            indices_batch = sample(range(len(counts)), self._num_classes)

            group_batch = [counts[ind][0] for ind in indices_batch]
            group_batches.append(group_batch)

            del_inds = []
            for ind in indices_batch:
                counts[ind][1] -= 1
                if counts[ind][1] == 0:
                    del_inds.append(ind)

            for ind in sorted(del_inds, reverse=True):
                del counts[ind]

        # Shuffle all labels within class, get extra samples to fill the batch
        class_to_labels = deepcopy(self._class_to_labels)
        for key, key_labels in class_to_labels.items():
            missing_items = (
                self._num_items_per_class - len(key_labels) % self._num_items_per_class
            )
            if missing_items != self._num_items_per_class:
                key_labels += choices(key_labels, k=missing_items)
            shuffle(key_labels)

        # Construct batches
        batches = []
        for group_batch in group_batches:
            batch = []
            for cls_group in group_batch:
                cls_items = class_to_labels[cls_group][-self._num_items_per_class :]
                del class_to_labels[cls_group][-self._num_items_per_class :]
                batch += cls_items

            batches.append(batch)

        self._batches = batches


class SessionSampler(BaseSampler):
    """A batch sampler that fills the batch with items with items from as many sessions
    as possible.

    When constructing each batch, the sampler will start by adding all items of one
    session, and continue adding sessions in this way until the total number of items
    in equals ``batch_size`` - from the last session in batch only the first few items
    are taken, so as to not exceed the desider batch size.

    The last session in batch, if it was not included completely and was not the only
    session in the batch, will appear again in next the batch as the first added session
    (with all items, including those that appeared in the previous batch).

    It is assumed that the anchor document is the first document in its session -
    something that is always true if labels come from :class:`SessionDataset`
    """

    def __init__(
        self, labels: Sequence[Tuple[int, int]], batch_size: int, shuffle: bool = True
    ):
        """Constuct the batch sampler.

        :param labels: A sequence of items labels, each label should be a tuple with two
            integers - first one is the id of the session, while the second one denots
            the match type of the item (0 for root document, 1 for positive match
            and -1 for negative match)
        :param batch_size: How many items to include in a batch
        :param shuffle: Suffle the order of sessions. If false, will use the order of
            sessions which is given by ``labels``.
        """

        self._batch_size = batch_size
        self._shuffle = shuffle
        # The locations and types of labels for each session
        self._sessions = defaultdict(list)
        for ind, (session, match_type) in enumerate(labels):
            self._sessions[session].append([ind, match_type])

        super().__init__(labels, batch_size)

    def _prepare_batches(self):

        # Set up the order of sessions
        sessions = list(self._sessions.keys())
        if self._shuffle:
            shuffle(sessions)

        batches = []
        current_batch = []

        for session in sessions:
            session_items = self._sessions[session]
            session_exhausted = False

            while not session_exhausted:
                if len(current_batch) + len(session_items) <= self._batch_size:
                    current_batch += [x[0] for x in session_items]
                    session_exhausted = True
                else:
                    missing = self._batch_size - len(current_batch)

                    # If the session is larger than batch size, use and go to next
                    if len(current_batch) == 0:
                        session_exhausted = True

                    # Take the first one (root), sample the rest
                    current_batch += [session_items[0][0]]
                    current_batch += [
                        x[0] for x in sample(session_items[1:], missing - 1)
                    ]

                if len(current_batch) == self._batch_size:
                    batches.append(current_batch)
                    current_batch = []

        # Add remainder batch
        if len(current_batch):
            batches.append(current_batch)

        self._batches = batches
