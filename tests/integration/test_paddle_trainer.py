# build a simple dense network with bottleneck as 10-dim
from paddle import nn

# wrap the user model with our trainer
from trainer.paddle import PaddleTrainer

# generate artificial positive & negative data
from ..data_generator import fashion_match_doc_generator as fmdg


def test_simple_sequential_model():
    user_model = nn.Sequential(
        nn.Flatten(start_axis=1),
        nn.Linear(in_features=784, out_features=128),
        nn.ReLU(),
        nn.Linear(in_features=128, out_features=32),
    )

    pt = PaddleTrainer(user_model, head_layer='CosineLayer')

    # fit and save the checkpoint
    pt.fit(fmdg(num_total=521), epochs=5, batch_size=256)
    from paddle.static import InputSpec

    x_spec = InputSpec(shape=[None, 28, 28], name='x')
    pt.save('./examples/fashion/trained', input_spec=[x_spec])
