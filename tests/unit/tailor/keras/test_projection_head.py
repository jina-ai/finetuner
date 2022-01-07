import pytest
import tensorflow as tf

from finetuner.tailor.keras import KerasTailor
from finetuner.tailor.keras.projection_head import ProjectionHead


@pytest.mark.parametrize(
    'in_features, output_dim, num_layers',
    [(2048, 128, 3), (2048, 256, 3), (1024, 512, 5)],
)
def test_projection_head(in_features, output_dim, num_layers):
    head = ProjectionHead(
        in_features=in_features, output_dim=output_dim, num_layers=num_layers
    )
    out = head(tf.random.uniform([2, in_features]))
    assert list(out.shape) == [2, output_dim]


def test_attach_custom_projection_head(tf_vgg16_cnn_model):
    def _create_bottleneck_model():
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=(4096,)))
        model.add(tf.keras.layers.Dense(1024, activation='relu'))
        model.add(tf.keras.layers.Dense(512, activation='softmax'))
        return model

    keras_tailor = KerasTailor(
        model=tf_vgg16_cnn_model,
        input_size=(224, 224, 3),
        input_dtype='float32',
    )
    tailed_model = keras_tailor.to_embedding_model(
        layer_name='fc1', freeze=False, projection_head=_create_bottleneck_model()
    )
    assert list(tailed_model.output.shape) == ([None, 512])


@pytest.mark.parametrize(
    'tf_model, input_size, input_, dim_projection_head, out_dim_embed_model',
    [
        ('tf_dense_model', (128,), (2, 128), 128, 10),
        ('tf_simple_cnn_model', (1, 28, 28), (2, 28, 28, 1), 128, 10),
        ('tf_vgg16_cnn_model', (224, 224, 3), (2, 224, 224, 3), 128, 1000),
        ('tf_stacked_lstm', (128,), (2, 128), 128, 5),
    ],
    indirect=['tf_model'],
)
def test_attach_default_projection_head(
    tf_model, input_size, input_, dim_projection_head, out_dim_embed_model
):
    keras_tailor = KerasTailor(model=tf_model, input_size=input_size)
    tailed_model = keras_tailor.to_embedding_model(
        freeze=False, projection_head=ProjectionHead(in_features=out_dim_embed_model)
    )
    rand_input = tf.constant(tf.random.uniform(input_))
    out = tailed_model(rand_input)
    assert list(out.shape) == [2, dim_projection_head]
