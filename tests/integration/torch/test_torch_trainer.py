import os

import torch
import torch.nn as nn
import numpy as np
import pytest

from trainer.pytorch import PytorchTrainer
from ...data_generator import fashion_match_doc_generator as fmdg


@pytest.mark.parametrize('head_layer', ['CosineLayer', 'TripletLayer'])
def test_simple_sequential_model(tmpdir, params, head_layer):
    user_model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(
            in_features=params['input_dim'] * params['input_dim'],
            out_features=params['feature_dim'],
        ),
        nn.ReLU(),
        nn.Linear(in_features=params['feature_dim'], out_features=params['output_dim']),
    )
    model_path = os.path.join(tmpdir, 'trained.pth')

    pt = PytorchTrainer(user_model, head_layer=head_layer)

    # fit and save the checkpoint
    pt.fit(
        lambda: fmdg(num_total=1000),
        epochs=params['epochs'],
        batch_size=params['batch_size'],
    )
    pt.save(model_path)

    # load the checkpoint and ensure the dim
    embedding_model = torch.load(model_path)
    embedding_model.eval()
    num_samples = 5
    inputs = torch.from_numpy(
        np.random.random(
            [num_samples, params['input_dim'], params['input_dim']]
        ).astype(np.float32)
    )
    r = embedding_model(inputs)
    assert r.shape == (num_samples, params['output_dim'])
