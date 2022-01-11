import numpy as np
import pytest
import tensorflow as tf

from finetuner.tailor.keras import KerasTailor


@pytest.mark.parametrize(
    'tf_model, layer_name',
    [
        ('tf_dense_model', 'random_name'),
        ('tf_simple_cnn_model', 'random_name'),
        ('tf_vgg16_cnn_model', 'random_name'),
        ('tf_stacked_lstm', 'random_name'),
        ('tf_bidirectional_lstm', 'random_name'),
    ],
    indirect=['tf_model'],
)
def test_trim_fail_given_unexpected_layer_name(tf_model, layer_name):
    with pytest.raises(KeyError):
        keras_tailor = KerasTailor(tf_model)
        keras_tailor.to_embedding_model(layer_name=layer_name)


@pytest.mark.parametrize(
    'tf_model, layer_name, expected_output_shape',
    [
        ('tf_dense_model', 'dense_3', (None, 10)),
        ('tf_simple_cnn_model', 'dropout_1', (None, 128)),
        ('tf_vgg16_cnn_model', 'fc2', (None, 4096)),
        ('tf_stacked_lstm', 'dense', (None, 256)),
        ('tf_bidirectional_lstm', 'dense', (None, 32)),
        ('tf_dense_model', None, (None, 10)),
        ('tf_simple_cnn_model', None, (None, 10)),
        ('tf_vgg16_cnn_model', None, (None, 1000)),
        ('tf_stacked_lstm', None, (None, 5)),
        ('tf_bidirectional_lstm', None, (None, 32)),
    ],
    indirect=['tf_model'],
)
def test_to_embedding_model(tf_model, layer_name, expected_output_shape):
    keras_tailor = KerasTailor(tf_model)
    model = keras_tailor.to_embedding_model(layer_name=layer_name)
    assert model.output_shape == expected_output_shape


def test_weights_preserved_given_pretrained_model(tf_vgg16_cnn_model):
    weights = tf_vgg16_cnn_model.layers[0].get_weights()
    keras_tailor = KerasTailor(tf_vgg16_cnn_model)
    vgg16_cnn_model = keras_tailor.to_embedding_model(layer_name='fc2')
    weights_after_convert = vgg16_cnn_model.layers[0].get_weights()
    np.testing.assert_array_equal(weights, weights_after_convert)


@pytest.mark.parametrize(
    'tf_model',
    [
        'tf_dense_model',
        'tf_simple_cnn_model',
        'tf_vgg16_cnn_model',
        'tf_stacked_lstm',
        'tf_bidirectional_lstm',
    ],
    indirect=['tf_model'],
)
@pytest.mark.parametrize('freeze', [True, False])
def test_freeze(tf_model, freeze):
    keras_tailor = KerasTailor(tf_model)
    for layer in tf_model.layers:
        assert layer.trainable
    model = keras_tailor.to_embedding_model(freeze=freeze)
    for idx, layer in enumerate(model.layers):
        if freeze:
            assert not layer.trainable
        else:
            assert layer.trainable


def test_freeze_given_bottleneck_model_and_freeze_is_true(tf_simple_cnn_model):
    def _create_bottleneck_model():
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=(128,)))
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        return model

    paddle_tailor = KerasTailor(
        model=tf_simple_cnn_model,
        input_size=(28, 28, 1),
        input_dtype='float32',
    )

    model = paddle_tailor.to_embedding_model(
        freeze=True, layer_name='dropout_1', projection_head=_create_bottleneck_model()
    )
    # assert bottleneck model is not freezed
    for layer in model.layers:
        if layer.name == 'dense_2':
            assert layer.trainable
        else:
            assert not layer.trainable


@pytest.mark.parametrize(
    'tf_model, layer_name, input_size, input_dtype, freeze_layers',
    [
        ('tf_dense_model', 10, (128,), 'float32', ['linear_1', 'linear_5']),
        ('tf_simple_cnn_model', 2, (1, 28, 28), 'float32', ['conv2d_1', 'maxpool2d_5']),
        (
            'tf_vgg16_cnn_model',
            4,
            (3, 224, 224),
            'float32',
            ['conv2d_27', 'maxpool2d_31', 'adaptiveavgpool2d_32'],
        ),
        ('tf_stacked_lstm', 10, (128,), 'int64', ['linear_layer_1', 'linear_layer_2']),
        ('tf_bidirectional_lstm', 5, (128,), 'int64', ['lastcell_3', 'linear_4']),
    ],
    indirect=['tf_model'],
)
def test_freeze_given_freeze_layers(
    tf_model, layer_name, input_size, input_dtype, freeze_layers
):
    pytorch_tailor = KerasTailor(
        model=tf_model,
        input_size=input_size,
        input_dtype=input_dtype,
    )
    model = pytorch_tailor.to_embedding_model(
        freeze=freeze_layers,
    )
    for layer, param in zip(pytorch_tailor.embedding_layers, model.layers):
        layer_name = layer['name']
        if layer_name in freeze_layers:
            assert not param.trainable
        else:
            assert param.trainable


def test_keras_model_parser():
    user_model = tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28), name='l1'),
            tf.keras.layers.Dense(128, activation='relu', name='l2'),
            tf.keras.layers.Dense(32, name='l3'),
        ]
    )

    keras_tailor = KerasTailor(user_model)

    r = keras_tailor.embedding_layers
    assert len(r) == 3
    assert r[0]['name'] == 'l1'
    assert r[1]['name'] == 'l2'
    assert r[2]['name'] == 'l3'

    assert r[0]['output_features'] == 784
    assert r[0]['nb_params'] == 0

    assert r[1]['output_features'] == 128
    assert r[1]['nb_params'] == 100480

    assert r[2]['output_features'] == 32
    assert r[2]['nb_params'] == 4128
