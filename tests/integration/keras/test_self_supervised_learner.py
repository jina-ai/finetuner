from functools import partial

import pytest

import finetuner as ft
from finetuner.tailor.keras.projection_head import ProjectionHead
from finetuner.tuner.augmentation import vision_preprocessor
from finetuner.tuner.keras.losses import NTXentLoss


@pytest.fixture
def default_model():
    import tensorflow as tf

    return tf.keras.applications.resnet50.ResNet50(weights=None)


@pytest.mark.parametrize(
    "n_cls,n_epochs,loss_cls,temperature",
    [
        (5, 2, NTXentLoss, 0.1),
        (10, 2, NTXentLoss, 0.2),
        (10, 5, NTXentLoss, 1.0),
    ],
)
def test_self_supervised_learning(
    default_model, create_easy_data_instance, n_cls, n_epochs, loss_cls, temperature
):
    # Prepare model and data
    data, vecs = create_easy_data_instance(n_cls)

    projection_head = ProjectionHead(in_features=2048)
    model = ft.fit(
        model=default_model,
        train_data=data,
        epochs=n_epochs,
        batch_size=len(data),
        loss=loss_cls(temperature),
        num_items_per_class=2,
        learning_rate=1e-2,
        preprocess_fn=partial(vision_preprocessor, target_channel_axis=-1),
        to_embedding_model=True,
        layer_name='avg_pool',
        projection_head=projection_head,
        input_size=(224, 224, 3),
    )
    assert model
