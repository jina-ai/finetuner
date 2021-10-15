import multiprocessing
import os
import random
import time

import pytest
import requests
from jina.helper import random_port

os.environ['JINA_LOG_LEVEL'] = 'DEBUG'
import paddle
import torch

from finetuner.toydata import generate_qa_match


class LastCellPT(torch.nn.Module):
    def forward(self, x):
        out, _ = x
        return out[:, -1, :]


class LastCellPD(paddle.nn.Layer):
    def forward(self, x):
        out, _ = x
        return out[:, -1, :]


def _run(framework_name, head_layer, port_expose):
    from finetuner import fit

    import paddle
    import tensorflow as tf
    import torch

    embed_models = {
        'keras': lambda: tf.keras.Sequential(
            [
                tf.keras.layers.Embedding(input_dim=5000, output_dim=64),
                tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
                tf.keras.layers.Dense(32),
            ]
        ),
        'pytorch': lambda: torch.nn.Sequential(
            torch.nn.Embedding(num_embeddings=5000, embedding_dim=64),
            torch.nn.LSTM(64, 64, bidirectional=True, batch_first=True),
            LastCellPT(),
            torch.nn.Linear(in_features=2 * 64, out_features=32),
        ),
        'paddle': lambda: paddle.nn.Sequential(
            paddle.nn.Embedding(num_embeddings=5000, embedding_dim=64),
            paddle.nn.LSTM(64, 64, direction='bidirectional'),
            LastCellPD(),
            paddle.nn.Linear(in_features=2 * 64, out_features=32),
        ),
    }

    fit(
        embed_models[framework_name](),
        generate_qa_match(num_total=10, num_neg=0),
        head_layer=head_layer,
        interactive=True,
        port_expose=port_expose,
    )


# 'keras' does not work under this test setup
# Exception ... ust be from the same graph as Tensor ...
# TODO: add keras backend back to the test
@pytest.mark.parametrize('framework', ['pytorch', 'paddle'])
@pytest.mark.parametrize('head_layer', ['CosineLayer', 'TripletLayer'])
def test_all_frameworks(framework, head_layer, tmpdir):
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

        # mimic label & fit
        for lbl_doc in rj['data']['docs']:
            for m in lbl_doc['matches']:
                m['finetuner'] = {'label': random.sample([-1, 1], 1)[0]}

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

    except:
        raise
    finally:
        p.terminate()
