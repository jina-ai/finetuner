import tensorflow as tf
from finetuner.tuner.keras import KerasTuner

from ..conftest import RecordCallback


def test_basic_callback(generate_random_data, expected_results):
    train_data = generate_random_data(8, 2, 2)
    val_data = generate_random_data(4, 2, 2)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10, activation="relu"),
        ]
    )

    # Train
    callback = RecordCallback()
    tuner = KerasTuner(model, callbacks=[callback])
    tuner.fit(
        train_data=train_data,
        eval_data=val_data,
        epochs=2,
        batch_size=4,
        num_items_per_class=2,
    )

    expected_calls = [x[0] for x in expected_results]
    expected_epochs = [x[1] for x in expected_results]
    expected_batch_idx = [x[2] for x in expected_results]
    expected_num_epochs = [x[3] for x in expected_results]
    expected_num_batches_train = [x[4] for x in expected_results]
    expected_num_batches_val = [x[5] for x in expected_results]

    assert callback.calls == expected_calls
    assert callback.epochs == expected_epochs
    assert callback.num_epochs == expected_num_epochs
    assert callback.batch_idx == expected_batch_idx
    assert callback.num_batches_train == expected_num_batches_train
    assert callback.num_batches_val == expected_num_batches_val
