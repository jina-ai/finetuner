import multiprocessing
import os
import random
import time

import pytest
import requests
from jina.helper import random_port

from finetuner.toydata import generate_fashion_match as fdg


os.environ['JINA_LOG_LEVEL'] = 'DEBUG'


def _run(framework_name, head_layer, port_expose):
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

    fit(
        embed_models[framework_name](),
        fdg,
        head_layer=head_layer,
        interactive=True,
        port_expose=port_expose,
    )


# 'keras' does not work under this test setup
# Exception ... ust be from the same graph as Tensor ...
# TODO: add keras backend back to the test
@pytest.mark.parametrize('framework', ['pytorch', 'paddle'])
@pytest.mark.parametrize('head_layer', ['CosineLayer', 'TripletLayer'])
def test_all_frameworks(framework, head_layer):
    port = random_port()
    p = multiprocessing.Process(
        target=_run,
        args=(
            framework,
            head_layer,
            port,
        ),
    )
    p.start()
    try:
        while True:
            try:
                req = requests.get(
                    f'http://localhost:{port}/docs',
                )
                assert req.status_code == 200
                break
            except:
                print('wait for ready...')
                time.sleep(2)

        # mimic next page
        req = requests.post(
            f'http://localhost:{port}/next',
            json={
                'data': [],
                'parameters': {'start': 0, 'end': 1, 'topk': 5, 'sample_size': 10},
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
