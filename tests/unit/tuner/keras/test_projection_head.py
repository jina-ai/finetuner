import pytest
import tensorflow as tf

from finetuner.tuner.keras import KerasTuner, _ProjectionHead


@pytest.mark.parametrize(
    'in_features, output_dim, num_layers',
    [(2048, 128, 3), (2048, 256, 3), (1024, 512, 5)],
)
def test_projection_head(in_features, output_dim, num_layers):
    head = _ProjectionHead(
        in_features=in_features, output_dim=output_dim, num_layers=num_layers
    )
    out = head(tf.random.uniform([2, in_features]))
    assert list(out.shape) == [2, output_dim]


@pytest.mark.parametrize(
    'tf_model, input_size, input_, dim_projection_head, dim_representation, input_dtype',
    [
        # ('tf_dense_model', (128,), (2, 128), 128, 10, 'float32'),
        ('torch_simple_cnn_model', (1, 28, 28), (2, 1, 28, 28), 128, 10, 'float32'), # TODO FIX EAGER TENSOR
        # ('torch_vgg16_cnn_model', (3, 224, 224), (2, 3, 224, 224), 128, 1000, 'float32'),
        # ('torch_stacked_lstm', 128, (2, 128), 128, 5, 'int32'),
    ],
    indirect=['tf_model'],
)
def test_attach_detach_projection_head(
    tf_model, input_size, input_, dim_projection_head, dim_representation, input_dtype
):
    keras_tuner = KerasTuner(
        embed_model=tf_model, input_size=input_size, input_dtype=input_dtype
    )
    keras_tuner._attach_projection_head()
    layer_names = [layer.__class__.__name__ for layer in keras_tuner.embed_model.layers]
    assert '_ProjectionHead' in layer_names
    rand_input = tf.random.uniform(shape=input_)
    out = keras_tuner.embed_model(rand_input)
    assert list(out.shape) == [2, dim_projection_head]
    # do the same thing as in `fit` function to remove projection head.
    embed_model = tf.keras.Sequential(keras_tuner.embed_model.layers[:-1])
    embed_model.build(input_)
    out = embed_model(rand_input)
    assert list(out.shape) == [2, dim_representation]
