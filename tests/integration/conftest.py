from copy import deepcopy
from time import perf_counter, sleep

import numpy as np
import pytest
from finetuner import __default_tag_key__
from finetuner.tuner.callback.base import BaseCallback
from jina import Document, DocumentArray


@pytest.fixture
def params():
    return {
        'input_dim': 28,
        'output_dim': 8,
        'epochs': 2,
        'batch_size': 256,
        'num_items_per_class': 32,
        'feature_dim': 32,
        'learning_rate': 0.01,
        'num_train': 1000,
        'num_eval': 1000,
        'num_predict': 100,
    }


@pytest.fixture
def create_easy_data_class():
    def create_easy_data_fn(n_cls: int, dim: int):
        """Creates a class dataset from random vectors.

        Works as follows:
        - for each class, create two random vectors - so that each one has a positive
            sample as well. This will create 2 * n_cls unique random vectors, from
            which we build the dataset

        Note that there is no relationship between these vectors - they are all randomly
        generated. The purpose of this dataset is to verify that over-parametrized
        models can properly separate (or bring together) these random vectors, thus
        confirming that our training method works.
        """
        # Fix random seed so we can debug on data, if needed
        rng = np.random.default_rng(42)

        # Create random class vectors
        rand_vecs = rng.uniform(size=(2 * n_cls, dim)).astype(np.float32)
        labels = []
        for i in range(n_cls):
            labels += [i, i]

        docs = []
        for vec, label in zip(rand_vecs, labels):
            docs.append(Document(blob=vec, tags={__default_tag_key__: label}))

        return docs, rand_vecs

    return create_easy_data_fn


@pytest.fixture
def create_easy_data_session():
    def create_easy_data_fn(n_cls: int, dim: int, n_sample: int):
        """Creates a session dataset from random vectors.

        Works as follows:
        - for each class, create two random vectors - so that each one has a positive
            sample as well. This will create 2 * n_cls unique random vectors, from
            which we build the dataset
        - loop over the dataset (if n_sample > 2 * n_cls documents will be repeated),
            and for each vector add its positive sample, and vectors from all other
            classes as a negative sample. This is important, as it assures that each
            vector will see all others in training

        In the end you will have a dataset of size n_samples, where each item has
        one positive sample and 2 * (n_cls - 1) negative samples.

        Note that there is no relationship between these vectors - they are all randomly
        generated. The purpose of this dataset is to verify that over-parametrized
        models can properly separate (or bring together) these random vectors, thus
        confirming that our training method works.
        """

        # Fix random seed so we can debug on data, if needed
        rng = np.random.default_rng(42)

        # Create random class vectors
        rand_vecs = rng.uniform(size=(2 * n_cls, dim)).astype(np.float32)

        # Generate anchor-pos-neg triplets
        triplets = DocumentArray()
        for i in range(n_sample):
            anchor_ind = i % (2 * n_cls)
            pos_ind = anchor_ind - 1 if anchor_ind % 2 == 1 else anchor_ind + 1

            d = Document(blob=rand_vecs[anchor_ind])
            d.matches.append(
                Document(blob=rand_vecs[pos_ind], tags={__default_tag_key__: 1})
            )

            neg_inds = [j for j in range(2 * n_cls) if j not in [anchor_ind, pos_ind]]
            for neg_ind in neg_inds:
                d.matches.append(
                    Document(
                        blob=rand_vecs[neg_ind],
                        tags={__default_tag_key__: -1},
                    )
                )

            triplets.append(d)
        return triplets, rand_vecs

    return create_easy_data_fn


@pytest.fixture
def record_callback():
    class RecordCallback(BaseCallback):
        def __init__(self):
            self.calls = []
            self.epochs = []
            self.batch_idx = []
            self.num_epochs = []
            self.num_batches_train = []
            self.num_batches_val = []
            self.num_batches_query = []
            self.num_batches_index = []
            self.eval_metrics = []
            self.learning_rates = []

        def _record(self, tuner):
            self.epochs.append(tuner.state.epoch)
            self.batch_idx.append(tuner.state.batch_index)
            self.num_epochs.append(tuner.state.num_epochs)
            self.num_batches_train.append(tuner.state.num_batches_train)
            self.num_batches_val.append(tuner.state.num_batches_val)
            self.num_batches_query.append(tuner.state.num_batches_query)
            self.num_batches_index.append(tuner.state.num_batches_index)
            self.eval_metrics.append(deepcopy(tuner.state.eval_metrics))
            self.learning_rates.append(deepcopy(tuner.state.learning_rates))

        def on_fit_begin(self, tuner):
            self.calls.append('on_fit_begin')
            self._record(tuner)

        def on_epoch_begin(self, tuner):
            self.calls.append('on_epoch_begin')
            self._record(tuner)

        def on_train_epoch_begin(self, tuner):
            self.calls.append('on_train_epoch_begin')
            self._record(tuner)

        def on_train_batch_begin(self, tuner):
            self.calls.append('on_train_batch_begin')
            self._record(tuner)

        def on_train_batch_end(self, tuner):
            self.calls.append('on_train_batch_end')
            self._record(tuner)

        def on_train_epoch_end(self, tuner):
            self.calls.append('on_train_epoch_end')
            self._record(tuner)

        def on_val_begin(self, tuner):
            self.calls.append('on_val_begin')
            self._record(tuner)

        def on_val_batch_begin(self, tuner):
            self.calls.append('on_val_batch_begin')
            self._record(tuner)

        def on_val_batch_end(self, tuner):
            self.calls.append('on_val_batch_end')
            self._record(tuner)

        def on_val_end(self, tuner):
            self.calls.append('on_val_end')
            self._record(tuner)

        def on_metrics_begin(self, tuner):
            self.calls.append('on_metrics_begin')
            self._record(tuner)

        def on_metrics_query_begin(self, tuner):
            self.calls.append('on_metrics_query_begin')
            self._record(tuner)

        def on_metrics_query_batch_begin(self, tuner):
            self.calls.append('on_metrics_query_batch_begin')
            self._record(tuner)

        def on_metrics_query_batch_end(self, tuner):
            self.calls.append('on_metrics_query_batch_end')
            self._record(tuner)

        def on_metrics_query_end(self, tuner):
            self.calls.append('on_metrics_query_end')
            self._record(tuner)

        def on_metrics_index_begin(self, tuner):
            self.calls.append('on_metrics_index_begin')
            self._record(tuner)

        def on_metrics_index_batch_begin(self, tuner):
            self.calls.append('on_metrics_index_batch_begin')
            self._record(tuner)

        def on_metrics_index_batch_end(self, tuner):
            self.calls.append('on_metrics_index_batch_end')
            self._record(tuner)

        def on_metrics_index_end(self, tuner):
            self.calls.append('on_metrics_index_end')
            self._record(tuner)

        def on_metrics_match_begin(self, tuner):
            self.calls.append('on_metrics_match_begin')
            self._record(tuner)

        def on_metrics_match_end(self, tuner):
            self.calls.append('on_metrics_match_end')
            self._record(tuner)

        def on_metrics_end(self, tuner):
            self.calls.append('on_metrics_end')
            self._record(tuner)

        def on_epoch_end(self, tuner):
            self.calls.append('on_epoch_end')
            self._record(tuner)

        def on_fit_end(self, tuner):
            self.calls.append('on_fit_end')
            self._record(tuner)

    return RecordCallback()


@pytest.fixture
def expected_results():
    """
    Expected results (call, epoch, batch, number of epochs and number of batches
    for train, eval, query and index) when doing 2 epochs, with 2 train and 1 eval/index/query batch
    """
    return [
        ('on_fit_begin', 0, 0, 2, 0, 0, 0, 0),
        ('on_epoch_begin', 0, 0, 2, 2, 0, 0, 0),
        ('on_train_epoch_begin', 0, 0, 2, 2, 0, 0, 0),
        ('on_train_batch_begin', 0, 0, 2, 2, 0, 0, 0),
        ('on_train_batch_end', 0, 0, 2, 2, 0, 0, 0),
        ('on_train_batch_begin', 0, 1, 2, 2, 0, 0, 0),
        ('on_train_batch_end', 0, 1, 2, 2, 0, 0, 0),
        ('on_train_epoch_end', 0, 1, 2, 2, 0, 0, 0),
        ('on_val_begin', 0, 0, 2, 2, 1, 0, 0),
        ('on_val_batch_begin', 0, 0, 2, 2, 1, 0, 0),
        ('on_val_batch_end', 0, 0, 2, 2, 1, 0, 0),
        ('on_val_end', 0, 0, 2, 2, 1, 0, 0),
        ('on_metrics_begin', 0, 0, 2, 2, 1, 0, 0),
        ('on_metrics_query_begin', 0, 0, 2, 2, 1, 1, 0),
        ('on_metrics_query_batch_begin', 0, 0, 2, 2, 1, 1, 0),
        ('on_metrics_query_batch_end', 0, 0, 2, 2, 1, 1, 0),
        ('on_metrics_query_end', 0, 0, 2, 2, 1, 1, 0),
        ('on_metrics_index_begin', 0, 0, 2, 2, 1, 1, 1),
        ('on_metrics_index_batch_begin', 0, 0, 2, 2, 1, 1, 1),
        ('on_metrics_index_batch_end', 0, 0, 2, 2, 1, 1, 1),
        ('on_metrics_index_end', 0, 0, 2, 2, 1, 1, 1),
        ('on_metrics_match_begin', 0, 0, 2, 2, 1, 1, 1),
        ('on_metrics_match_end', 0, 0, 2, 2, 1, 1, 1),
        ('on_metrics_end', 0, 0, 2, 2, 1, 1, 1),
        ('on_epoch_end', 0, 0, 2, 2, 1, 1, 1),
        ('on_epoch_begin', 1, 0, 2, 2, 1, 1, 1),
        ('on_train_epoch_begin', 1, 0, 2, 2, 1, 1, 1),
        ('on_train_batch_begin', 1, 0, 2, 2, 1, 1, 1),
        ('on_train_batch_end', 1, 0, 2, 2, 1, 1, 1),
        ('on_train_batch_begin', 1, 1, 2, 2, 1, 1, 1),
        ('on_train_batch_end', 1, 1, 2, 2, 1, 1, 1),
        ('on_train_epoch_end', 1, 1, 2, 2, 1, 1, 1),
        ('on_val_begin', 1, 0, 2, 2, 1, 1, 1),
        ('on_val_batch_begin', 1, 0, 2, 2, 1, 1, 1),
        ('on_val_batch_end', 1, 0, 2, 2, 1, 1, 1),
        ('on_val_end', 1, 0, 2, 2, 1, 1, 1),
        ('on_metrics_begin', 1, 0, 2, 2, 1, 1, 1),
        ('on_metrics_query_begin', 1, 0, 2, 2, 1, 1, 1),
        ('on_metrics_query_batch_begin', 1, 0, 2, 2, 1, 1, 1),
        ('on_metrics_query_batch_end', 1, 0, 2, 2, 1, 1, 1),
        ('on_metrics_query_end', 1, 0, 2, 2, 1, 1, 1),
        ('on_metrics_index_begin', 1, 0, 2, 2, 1, 1, 1),
        ('on_metrics_index_batch_begin', 1, 0, 2, 2, 1, 1, 1),
        ('on_metrics_index_batch_end', 1, 0, 2, 2, 1, 1, 1),
        ('on_metrics_index_end', 1, 0, 2, 2, 1, 1, 1),
        ('on_metrics_match_begin', 1, 0, 2, 2, 1, 1, 1),
        ('on_metrics_match_end', 1, 0, 2, 2, 1, 1, 1),
        ('on_metrics_end', 1, 0, 2, 2, 1, 1, 1),
        ('on_epoch_end', 1, 0, 2, 2, 1, 1, 1),
        ('on_fit_end', 1, 0, 2, 2, 1, 1, 1),
    ]


@pytest.fixture
def exception_callback():
    class ExceptionCallback(BaseCallback):
        def __init__(self, exception):
            self.exception = exception
            self.calls = []

        def on_fit_begin(self, tuner):
            raise self.exception

        def on_exception(self, tuner, exception):
            self.calls.append('on_exception')

        def on_keyboard_interrupt(self, tuner):
            self.calls.append('on_keyboard_interrupt')

    return ExceptionCallback


@pytest.fixture
def results_lr():
    """
    Get recorded learning rates for the exponential scheduler test
    """

    def get_results(scheduler_step):
        if scheduler_step == 'batch':
            return [
                None,
                None,
                None,
                1,
                1,
                1e-1,
                1e-1,
                1e-1,
                1e-1,
                1e-1,
                1e-1,
                1e-2,
                1e-2,
                1e-3,
                1e-3,
                1e-3,
                1e-3,
                1e-3,
            ]
        elif scheduler_step == 'epoch':
            return [
                None,
                None,
                None,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1e-1,
                1e-1,
                1e-1,
                1e-1,
                1e-1,
                1e-1,
                1e-1,
            ]

    return get_results


@pytest.fixture
def multi_workers_callback():
    """
    Used to time training and artifically set each bath to last 1 second,
    to check that multi-process loading works well.
    """

    class TrainingTimer(BaseCallback):
        def __init__(self):
            self.batch_times = []
            self._start = None

        def on_train_batch_begin(self, tuner):
            if self._start:
                self.batch_times.append(perf_counter() - self._start)
            self._start = perf_counter()
            sleep(1)  # "Training" should take exactly one second

    return TrainingTimer()


@pytest.fixture
def multi_workers_preprocess_fn():
    """
    A preprocessing function that delays preprocessing of each item by one second
    """

    def preprocess_fn(d: Document):
        sleep(1)

        return d.content

    return preprocess_fn
