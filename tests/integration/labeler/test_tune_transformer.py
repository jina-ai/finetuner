import multiprocessing
import os
import random
import time
from typing import List

import pytest
import requests
from jina.helper import random_port
from transformers import AutoModel
from transformers import AutoTokenizer

os.environ['JINA_LOG_LEVEL'] = 'DEBUG'
import torch

from finetuner.toydata import generate_qa
from finetuner import __default_tag_key__

TRANSFORMER_MODEL = 'sentence-transformers/paraphrase-MiniLM-L6-v2'


class TransformerEmbedder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained(TRANSFORMER_MODEL)

    def forward(self, inputs):
        out_model = self.model(**inputs)
        cls_token = out_model.last_hidden_state[:, 0, :]
        return cls_token


def _run(loss, port_expose):
    from finetuner import fit

    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL)

    def collate_fn(inputs: List[str]):
        batch_tokens = tokenizer(
            inputs,
            truncation=True,
            max_length=50,
            padding=True,
            return_tensors='pt',
        )
        return batch_tokens

    rv1, rv2 = fit(
        TransformerEmbedder(),
        generate_qa(num_total=10, num_neg=0),
        loss=loss,
        interactive=True,
        port_expose=port_expose,
        collate_fn=collate_fn,
    )

    assert rv1
    assert not rv2


all_test_losses = ['SiameseLoss', 'TripletLoss']


@pytest.mark.parametrize('loss', all_test_losses)
def test_all_frameworks(loss, tmpdir):
    port = random_port()
    p = multiprocessing.Process(
        target=_run,
        args=(loss, port),
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
        assert os.path.isfile(model_path)

    except:
        raise
    finally:
        p.terminate()
