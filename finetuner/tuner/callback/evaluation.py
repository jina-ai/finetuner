import math
from typing import TYPE_CHECKING, Optional

from ... import embed
from ..dataset.helper import batch_document_sequence
from ..evaluation import Evaluator
from .base import BaseCallback

if TYPE_CHECKING:
    from ...helper import DocumentSequence
    from ..base import BaseTuner


class EvaluationCallback(BaseCallback):
    """
    A callback that uses the Evaluator to calculate IR metrics at the end of each epoch. When used
    with other callbacks that rely on metrics, like checkpoints and logging, this callback should be
    defined first, so that it precedes in execution.
    """

    def __init__(
        self,
        query_data: 'DocumentSequence',
        index_data: Optional['DocumentSequence'] = None,
        limit: int = 20,
        distance: str = 'cosine',
        num_workers: int = 1,
    ):
        """
        :param query_data: Search data used by the evaluator at the end of each epoch, to evaluate the model.
        :param index_data: Index data or catalog used by the evaluator at the end of each epoch, to evaluate the model.
        :param limit: The number of top search results to consider, when computing the evaluation metrics.
        :param distance: The type of distance metric to use when matching query and index docs, available options are
            ``'cosine'``, ``'euclidean'`` and ``'sqeuclidean'``.
        :param num_workers: The number of workers to use when matching query and index data.
        """
        self._query_data = query_data
        self._index_data = index_data
        self._limit = limit
        self._distance = distance
        self._num_workers = num_workers
        self._query_pbar_id = None
        self._index_pbar_id = None
        self._match_pbar_id = None

    def on_fit_begin(self, tuner: 'BaseTuner'):
        self._query_pbar_id = tuner._progress_bar.add_task(
            'Embedding queries', visible=False, start=False
        )
        self._index_pbar_id = tuner._progress_bar.add_task(
            'Embedding index', visible=False, start=False
        )
        self._match_pbar_id = tuner._progress_bar.add_task(
            'Matching', visible=False, start=False
        )

    def on_epoch_end(self, tuner: 'BaseTuner'):

        # start query data progress bar
        tuner._progress_bar.reset(
            self._query_pbar_id,
            visible=True,
            description='Embedding queries',
            total=math.ceil(len(self._query_data) / tuner._batch_size),
            completed=0,
            metrics='',
        )

        # embed queries
        for _, batch in enumerate(
            batch_document_sequence(self._query_data, size=tuner._batch_size)
        ):
            embed(
                batch,
                tuner._embed_model,
                device=tuner._device_name,
                batch_size=tuner._batch_size,
                preprocess_fn=tuner._preprocess_fn,
                collate_fn=tuner._collate_fn,
            )
            tuner._progress_bar.update(task_id=self._query_pbar_id, advance=1)
        tuner._progress_bar.update(task_id=self._query_pbar_id, visible=False)

        if self._index_data:

            # start index data progress bar
            tuner._progress_bar.reset(
                self._index_pbar_id,
                visible=True,
                description='Embedding index',
                total=math.ceil(len(self._index_data) / tuner._batch_size),
                completed=0,
                metrics='',
            )

            # embed index
            for _, batch in enumerate(
                batch_document_sequence(self._index_data, size=tuner._batch_size)
            ):
                embed(
                    batch,
                    tuner._embed_model,
                    device=tuner._device_name,
                    batch_size=tuner._batch_size,
                    preprocess_fn=tuner._preprocess_fn,
                    collate_fn=tuner._collate_fn,
                )
                tuner._progress_bar.update(task_id=self._index_pbar_id, advance=1)

            index_data = self._index_data
            tuner._progress_bar.update(task_id=self._index_pbar_id, visible=False)

        else:
            index_data = self._query_data

        # start matching progress bar
        tuner._progress_bar.reset(
            self._match_pbar_id,
            visible=True,
            description='Matching',
            metrics='',
        )

        # compute metrics
        evaluator = Evaluator(self._query_data, index_data)
        tuner.state.eval_metrics = evaluator.evaluate(
            limit=self._limit,
            distance=self._distance,
            label=f'epoch#{tuner.state.epoch}',
            num_workers=self._num_workers,
        )
        tuner._progress_bar.update(task_id=self._match_pbar_id, visible=False)
