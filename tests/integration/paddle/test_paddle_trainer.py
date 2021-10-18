import numpy as np
import paddle
import pytest
from paddle import nn

from finetuner.tuner import fit, save
from finetuner.toydata import generate_fashion_match
from finetuner.toydata import generate_qa_match


@pytest.mark.parametrize(
    'loss',
    [
        'CosineSiameseLoss',
        'EuclideanSiameseLoss',
        'EuclideanTripletLoss',
        'CosineTripletLoss',
    ],
)
def test_simple_sequential_model(tmpdir, params, loss):
    user_model = nn.Sequential(
        nn.Flatten(start_axis=1),
        nn.Linear(
            in_features=params['input_dim'] * params['input_dim'],
            out_features=params['feature_dim'],
        ),
        nn.ReLU(),
        nn.Linear(in_features=params['feature_dim'], out_features=params['output_dim']),
    )

    model_path = tmpdir / 'trained.pd'
    # fit and save the checkpoint
    train_data, train_catalog = generate_fashion_match(
        num_neg=10, num_pos=10, num_total=params['num_train'], pre_init_generator=False
    )
    eval_data, eval_catalog = generate_fashion_match(
        num_neg=10,
        num_pos=10,
        num_total=params['num_eval'],
        is_testset=True,
        pre_init_generator=False,
    )
    train_catalog.extend(eval_catalog)

    fit(
        user_model,
        loss=loss,
        train_data=train_data,
        eval_data=eval_data,
        catalog=train_catalog,
        epochs=params['epochs'],
        batch_size=params['batch_size'],
    )

    save(user_model, model_path)

    user_model.set_state_dict(paddle.load(model_path))
    user_model.eval()
    inputs = paddle.to_tensor(
        np.random.random(
            [params['num_predict'], params['input_dim'], params['input_dim']]
        ).astype(np.float32)
    )
    r = user_model(inputs)
    assert tuple(r.shape) == (params['num_predict'], params['output_dim'])


@pytest.mark.parametrize(
    'loss',
    [
        'CosineSiameseLoss',
        'EuclideanSiameseLoss',
        'EuclideanTripletLoss',
        'CosineTripletLoss',
    ],
)
def test_simple_lstm_model(tmpdir, params, loss):
    class extractlastcell(nn.Layer):
        def forward(self, x):
            out, _ = x
            return out[:, -1, :]

    user_model = nn.Sequential(
        nn.Embedding(num_embeddings=5000, embedding_dim=params['feature_dim']),
        nn.LSTM(
            params['feature_dim'],
            params['feature_dim'],
            direction='bidirectional',
        ),
        extractlastcell(),
        nn.Linear(
            in_features=2 * params['feature_dim'], out_features=params['output_dim']
        ),
    )
    model_path = tmpdir / 'trained.pd'

    # fit and save the checkpoint
    train_data, train_catalog = generate_qa_match(
        num_total=params['num_train'],
        max_seq_len=params['max_seq_len'],
        num_neg=5,
        is_testset=False,
        pre_init_generator=False,
    )
    eval_data, eval_catalog = generate_qa_match(
        num_total=params['num_train'],
        max_seq_len=params['max_seq_len'],
        num_neg=5,
        is_testset=True,
        pre_init_generator=False,
    )
    train_catalog.extend(eval_catalog)

    fit(
        user_model,
        loss=loss,
        train_data=train_data,
        eval_data=eval_data,
        catalog=train_catalog,
        epochs=params['epochs'],
        batch_size=params['batch_size'],
    )
    save(user_model, model_path)

    # load the checkpoint and ensure the dim
    user_model.set_state_dict(paddle.load(model_path))
    user_model.eval()
    inputs = paddle.to_tensor(
        np.random.randint(
            low=0,
            high=100,
            size=[params['num_predict'], params['max_seq_len']],
        ).astype(np.long)
    )
    r = user_model(inputs)
    assert tuple(r.shape) == (params['num_predict'], params['output_dim'])
