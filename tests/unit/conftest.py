import numpy as np
import pytest
from jina import Document, DocumentArray

from finetuner import __default_tag_key__


@pytest.fixture
def generate_random_triplets():
    """Returns a function that produces triplets with random embeddings."""

    def gen_fn(n: int, dim: int):
        rand_vecs = np.random.rand(n * 3, dim).astype(np.float32)

        # Generate anchor-pos-neg triplets
        triplets = DocumentArray()
        for i in range(n):
            d = Document(blob=rand_vecs[i * 3])
            d.matches.extend(
                [
                    Document(
                        blob=rand_vecs[i * 3 + 1],
                        tags={__default_tag_key__: {'label': 1}},
                    ),
                    Document(
                        blob=rand_vecs[i * 3 + 2],
                        tags={__default_tag_key__: {'label': -1}},
                    ),
                ]
            )

            triplets.append(d)
        return triplets

    return gen_fn
