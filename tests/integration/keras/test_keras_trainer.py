import os

import numpy as np
import pytest
import tensorflow as tf
from tensorflow import keras

from finetuner.toydata import generate_fashion
from finetuner.tuner.keras import KerasTuner


@pytest.mark.parametrize("loss", ["TripletLoss", "SiameseLoss"])
def test_simple_sequential_model(tmpdir, params, loss):
    user_model = tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(
                input_shape=(params["input_dim"], params["input_dim"])
            ),
            tf.keras.layers.Dense(params["feature_dim"], activation="relu"),
            tf.keras.layers.Dense(
                params["output_dim"],
            ),
        ]
    )
    model_path = os.path.join(tmpdir, "trained.kt")

    kt = KerasTuner(user_model, loss=loss)

    # fit and save the checkpoint
    kt.fit(
        train_data=generate_fashion(num_total=params["num_train"]),
        eval_data=generate_fashion(is_testset=True, num_total=params["num_eval"]),
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        num_items_per_class=params["num_items_per_class"],
    )
    kt.save(model_path)

    # load the checkpoint and ensure the dim
    embedding_model = keras.models.load_model(model_path)
    r = embedding_model.predict(
        np.random.random(
            [params["num_predict"], params["input_dim"], params["input_dim"]]
        )
    )
    assert r.shape == (params["num_predict"], params["output_dim"])


@pytest.mark.parametrize("loss", ["TripletLoss", "SiameseLoss"])
def test_session_data(loss, create_easy_data_session):
    """Test with session dataset"""

    # Prepare model and data
    data, _ = create_easy_data_session(5, 10, 2)

    # Simple model
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10, activation="relu"),
        ]
    )

    # Train
    tuner = KerasTuner(model, loss=loss)
    tuner.fit(train_data=data, epochs=2, batch_size=12)


def test_custom_optimizer(create_easy_data_session):
    """Test training using a custom optimizer"""

    # Prepare model and data
    data, _ = create_easy_data_session(5, 10, 2)

    # Simple model
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10, activation="relu"),
        ]
    )

    optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)

    # Train
    tuner = KerasTuner(model, loss="TripletLoss")
    tuner.fit(train_data=data, epochs=2, batch_size=10, optimizer=optimizer)
