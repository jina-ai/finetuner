import multiprocessing
import os
import random
import time

import pytest
import requests
from finetuner.toydata import generate_fashion
from finetuner import __default_tag_key__
from jina.helper import random_port

os.environ['JINA_LOG_LEVEL'] = 'DEBUG'

all_test_losses = ['SiameseLoss', 'TripletLoss']


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

    rv1, rv2 = fit(
        embed_models[framework_name](),
        generate_fashion(num_total=10),
        loss=loss,
        interactive=True,
        port_expose=port_expose,
    )

    assert rv1
    assert not rv2


# 'keras' does not work under this test setup
# Exception ... ust be from the same graph as Tensor ...
# TODO: add keras backend back to the test
@pytest.mark.parametrize('framework', ['pytorch', 'paddle'])
@pytest.mark.parametrize('loss', all_test_losses)
def test_all_frameworks(framework, loss, tmpdir):
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
                            'start': 0,
                            'end': 1,
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
                m[__default_tag_key__] = random.sample([-1, 1], 1)[0]

        req = requests.post(
            f'http://localhost:{port}/fit',
            json={'data': rj['data']['docs'], 'parameters': {'epochs': 10}},
        )
        assert req.status_code == 200

        model_path = os.path.join(tmpdir, 'model.train')
        req = requests.post(
            f'http://localhost:{port}/save',
            json={
                'data': [],
                'parameters': {'model_path': model_path},
            },
        )
        assert req.status_code == 200
        assert os.path.isfile(model_path)

        req = requests.post(
            f'http://localhost:{port}/terminate',
            json={
                'data': [],
                'parameters': {},
            },
        )
        assert req.status_code == 200

    except:
        raise
    finally:
        p.terminate()
