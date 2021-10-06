import os

import numpy as np
import pytest
import torch
import torch.nn as nn

from finetuner.tuner.pytorch import PytorchTuner
from finetuner.toydata import generate_fashion_match
from finetuner.toydata import generate_qa_match


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

    pt = PytorchTuner(user_model, head_layer=head_layer)

    # fit and save the checkpoint
    pt.fit(
        train_data=lambda: generate_fashion_match(
            num_pos=10, num_neg=10, num_total=params['num_train']
        ),
        eval_data=lambda: generate_fashion_match(
            num_pos=10, num_neg=10, num_total=params['num_eval'], is_testset=True
        ),
        epochs=params['epochs'],
        batch_size=params['batch_size'],
    )
    pt.save(model_path)

    # load the checkpoint and ensure the dim
    user_model.load_state_dict(torch.load(model_path))
    user_model.eval()
    inputs = torch.from_numpy(
        np.random.random(
            [params['num_predict'], params['input_dim'], params['input_dim']]
        ).astype(np.float32)
    )
    r = user_model(inputs)
    assert r.shape == (params['num_predict'], params['output_dim'])


@pytest.mark.parametrize('head_layer', ['CosineLayer', 'TripletLayer'])
def test_simple_lstm_model(tmpdir, params, head_layer):
    class extractlastcell(nn.Module):
        def forward(self, x):
            out, _ = x
            return out[:, -1, :]

    user_model = nn.Sequential(
        nn.Embedding(num_embeddings=5000, embedding_dim=params['feature_dim']),
        nn.LSTM(
            params['feature_dim'],
            params['feature_dim'],
            bidirectional=True,
            batch_first=True,
        ),
        extractlastcell(),
        nn.Linear(
            in_features=2 * params['feature_dim'], out_features=params['output_dim']
        ),
    )
    model_path = os.path.join(tmpdir, 'trained.pth')

    pt = PytorchTuner(user_model, head_layer=head_layer)

    # fit and save the checkpoint
    pt.fit(
        train_data=lambda: generate_qa_match(
            num_total=params['num_train'],
            max_seq_len=params['max_seq_len'],
            num_neg=5,
            is_testset=False,
        ),
        eval_data=lambda: generate_qa_match(
            num_total=params['num_eval'],
            max_seq_len=params['max_seq_len'],
            num_neg=5,
            is_testset=True,
        ),
        epochs=params['epochs'],
        batch_size=params['batch_size'],
    )
    pt.save(model_path)

    # load the checkpoint and ensure the dim
    user_model.load_state_dict(torch.load(model_path))
    user_model.eval()
    inputs = torch.from_numpy(
        np.random.randint(
            low=0,
            high=100,
            size=[params['num_predict'], params['max_seq_len']],
        ).astype(np.long)
    )
    r = user_model(inputs)
    assert r.shape == (params['num_predict'], params['output_dim'])
