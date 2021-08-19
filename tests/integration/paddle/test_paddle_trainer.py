import numpy as np
import paddle
from paddle import nn
from paddle.static import InputSpec

from trainer.paddle import PaddleTrainer
from ..data_generator import fashion_match_doc_generator as fmdg

INPUT_DIM = 28
OUTPUT_DIM = 32


def test_simple_sequential_model(tmpdir):
    user_model = nn.Sequential(
        nn.Flatten(start_axis=1),
        nn.Linear(in_features=INPUT_DIM * INPUT_DIM, out_features=128),
        nn.ReLU(),
        nn.Linear(in_features=128, out_features=OUTPUT_DIM),
    )

    pt = PaddleTrainer(user_model, head_layer='CosineLayer')

    # fit and save the checkpoint
    pt.fit(lambda: fmdg(num_total=1000), epochs=5, batch_size=256)

    x_spec = InputSpec(shape=[None, INPUT_DIM, INPUT_DIM])
    pt.save(tmpdir / 'trained.pd', input_spec=[x_spec])

    num_samples = 100
    embedding_model = paddle.jit.load(tmpdir / 'trained.pd')
    embedding_model.eval()
    r = embedding_model(
        paddle.to_tensor(np.random.random([num_samples, 28, 28]).astype(np.float32))
    )
    assert tuple(r.shape) == (num_samples, OUTPUT_DIM)
