import numpy as np
import paddle
import pytest
from paddle import nn
from paddle.static import InputSpec

from trainer.paddle import PaddleTrainer
from ...data_generator import fashion_match_doc_generator as fmdg


@pytest.mark.parametrize('head_layer', ['TripletLayer'])
def test_simple_sequential_model(tmpdir, params, head_layer):
    user_model = nn.Sequential(
        nn.Flatten(start_axis=1),
        nn.Linear(
            in_features=params['input_dim'] * params['input_dim'],
            out_features=params['feature_dim'],
        ),
        nn.ReLU(),
        nn.Linear(in_features=params['feature_dim'], out_features=params['output_dim']),
    )

    pt = PaddleTrainer(user_model, head_layer=head_layer)

    # fit and save the checkpoint
    pt.fit(
        lambda: fmdg(num_total=params['num_train']),
        epochs=params['epochs'],
        batch_size=params['batch_size'],
    )

    x_spec = InputSpec(shape=[None, params['input_dim'], params['input_dim']])
    pt.save(tmpdir / 'trained.pd', input_spec=[x_spec])

    embedding_model = paddle.jit.load(tmpdir / 'trained.pd')
    embedding_model.eval()
    r = embedding_model(
        paddle.to_tensor(
            np.random.random(
                [params['num_predict'], params['input_dim'], params['input_dim']]
            ).astype(np.float32)
        )
    )
    assert tuple(r.shape) == (params['num_predict'], params['output_dim'])
