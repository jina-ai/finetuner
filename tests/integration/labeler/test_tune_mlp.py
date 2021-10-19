import multiprocessing
import os
import random
import time

import pytest
import requests
from finetuner.toydata import generate_fashion_match_catalog
from jina.helper import random_port

os.environ['JINA_LOG_LEVEL'] = 'DEBUG'

all_test_losses = [
    'CosineSiameseLoss',
    'CosineTripletLoss',
    'EuclideanSiameseLoss',
    'EuclideanTripletLoss',
]


def _run(framework_name, loss, port_expose):
    from finetuner import fit

    import paddle
    import tensorflow as tf
    import torch

    embed_models = {
        'keras': lambda: tf.keras.Sequential(
            [
                tf.keras.layers.Flatten(input_shape=(28, 28)),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(32),
            ]
        ),
        'pytorch': lambda: torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(
                in_features=28 * 28,
                out_features=128,
            ),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=128, out_features=32),
        ),
        'paddle': lambda: paddle.nn.Sequential(
            paddle.nn.Flatten(),
            paddle.nn.Linear(
                in_features=28 * 28,
                out_features=128,
            ),
            paddle.nn.ReLU(),
            paddle.nn.Linear(in_features=128, out_features=32),
        ),
    }
    data, catalog = generate_fashion_match_catalog(
        num_total=10, num_catalog=100, num_pos=0, num_neg=0
    )

    fit(
        embed_models[framework_name](),
        data,
        catalog=catalog,
        loss=loss,
        interactive=True,
        port_expose=port_expose,
    )


# 'keras' does not work under this test setup
# Exception ... ust be from the same graph as Tensor ...
# TODO: add keras backend back to the test
@pytest.mark.parametrize('framework', ['pytorch', 'paddle'])
@pytest.mark.parametrize('loss', all_test_losses)
def test_all_frameworks(framework, loss):
    port = random_port()
    p = multiprocessing.Process(
        target=_run,
        args=(
            framework,
            loss,
            port,
        ),
    )
    p.start()
    try:
        while True:
            try:
                req = requests.post(
                    f'http://localhost:{port}/next',
                    json={
                        'data': [],
                        'parameters': {
                            'new_examples': 1,
                            'topk': 5,
                            'sample_size': 10,
                        },
                    },
                )
                assert req.status_code == 200
                assert req.json()['data']['docs']
                break
            except:
                print('wait for ready...')
                time.sleep(2)

        # mimic next page
        req = requests.post(
            f'http://localhost:{port}/next',
            json={
                'data': [],
                'parameters': {'new_examples': 1, 'topk': 5, 'sample_size': 10},
            },
        )
        assert req.status_code == 200
        rj = req.json()
        assert len(rj['data']['docs']) == 1
        assert len(rj['data']['docs'][0]['matches']) >= 4

        time.sleep(1)
        print('test fit...')
        # mimic label & fit
        for lbl_doc in rj['data']['docs']:
            for m in lbl_doc['matches']:
                m['finetuner'] = {'label': random.sample([-1, 1], 1)[0]}

        req = requests.post(
            f'http://localhost:{port}/fit',
            json={'data': rj['data']['docs'], 'parameters': {'epochs': 10}},
        )
        assert req.status_code == 200
    except:
        raise
    finally:
        p.terminate()
