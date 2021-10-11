import numpy as np
import pytest
from jina import Document, DocumentArray

from finetuner import __default_tag_key__


@pytest.fixture
def params():
    return {
        'input_dim': 28,
        'output_dim': 8,
        'epochs': 2,
        'batch_size': 256,
        'feature_dim': 32,
        'learning_rate': 0.01,
        'num_train': 1000,
        'num_eval': 1000,
        'num_predict': 100,
        'max_seq_len': 10,
    }


@pytest.fixture
def create_easy_data():
    def create_easy_data_fn(n_cls: int, dim: int, n_sample: int):
        """Creates a dataset from random vectors.

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
                Document(
                    blob=rand_vecs[pos_ind], tags={__default_tag_key__: {'label': 1}}
                )
            )

            neg_inds = [j for j in range(2 * n_cls) if j not in [anchor_ind, pos_ind]]
            for neg_ind in neg_inds:
                d.matches.append(
                    Document(
                        blob=rand_vecs[neg_ind],
                        tags={__default_tag_key__: {'label': -1}},
                    )
                )

            triplets.append(d)
        return triplets, rand_vecs

    return create_easy_data_fn
