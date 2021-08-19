import numpy as np
import paddle
from paddle import nn
from paddle.static import InputSpec

from trainer.paddle import PaddleTrainer
from ...data_generator import fashion_match_doc_generator as fmdg


def test_simple_sequential_model(tmpdir, params):
    user_model = nn.Sequential(
        nn.Flatten(start_axis=1),
        nn.Linear(
            in_features=params['input_dim'] * params['input_dim'],
            out_features=params['feature_dim'],
        ),
        nn.ReLU(),
        nn.Linear(in_features=params['feature_dim'], out_features=params['output_dim']),
    )

    pt = PaddleTrainer(user_model, head_layer='CosineLayer')

    # fit and save the checkpoint
    pt.fit(
        lambda: fmdg(num_total=1000),
        epochs=params['epochs'],
        batch_size=params['batch_size'],
    )

    x_spec = InputSpec(shape=[None, params['input_dim'], params['input_dim']])
    pt.save(tmpdir / 'trained.pd', input_spec=[x_spec])

    num_samples = 100
    embedding_model = paddle.jit.load(tmpdir / 'trained.pd')
    embedding_model.eval()
    r = embedding_model(
        paddle.to_tensor(
            np.random.random(
                [num_samples, params['input_dim'], params['input_dim']]
            ).astype(np.float32)
        )
    )
    assert tuple(r.shape) == (num_samples, params['output_dim'])
