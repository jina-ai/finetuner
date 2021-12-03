import os

import numpy as np
import paddle
import paddle.nn as nn
import pytest

from finetuner.toydata import generate_fashion
from finetuner.tuner.paddle import PaddleTuner


@pytest.mark.parametrize('loss', ['TripletLoss', 'SiameseLoss'])
def test_simple_sequential_model(tmpdir, params, loss):
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

    pt = PaddleTuner(user_model, loss=loss)

    # fit and save the checkpoint
    pt.fit(
        train_data=generate_fashion(num_total=params['num_train']),
        eval_data=generate_fashion(is_testset=True, num_total=params['num_eval']),
        epochs=params['epochs'],
        batch_size=params['batch_size'],
        num_items_per_class=params['num_items_per_class'],
    )
    pt.save(path=model_path)

    # load the checkpoint and ensure the dim
    user_model.set_state_dict(paddle.load(model_path)['state_dict'])
    user_model.eval()
    inputs = paddle.to_tensor(
        np.random.random(
            [params['num_predict'], params['input_dim'], params['input_dim']]
        ).astype(np.float32)
    )
    r = user_model(inputs)
    assert r.shape == [params['num_predict'], params['output_dim']]


@pytest.mark.parametrize('loss', ['TripletLoss', 'SiameseLoss'])
def test_session_data(loss, create_easy_data_session):
    """Test with session dataset"""

    # Prepare model and data
    data, _ = create_easy_data_session(5, 10, 2)

    # Simple model
    model = nn.Sequential(nn.Flatten(), nn.Linear(in_features=10, out_features=10))

    # Train
    tuner = PaddleTuner(model, loss=loss)
    tuner.fit(train_data=data, epochs=2, batch_size=12)
