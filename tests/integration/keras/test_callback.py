import pytest
import tensorflow as tf
from finetuner.tuner.keras import KerasTuner


def test_basic_callback(generate_random_data, expected_results, record_callback):
    train_data = generate_random_data(8, 2, 2)
    query_data = generate_random_data(4, 2, 2)
    index_data = generate_random_data(4, 2, 2)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10, activation='relu'),
        ]
    )

    # Train
    tuner = KerasTuner(model, callbacks=[record_callback])
    tuner.fit(
        train_data=train_data,
        query_data=query_data,
        index_data=index_data,
        epochs=2,
        batch_size=4,
        num_items_per_class=2,
    )

    expected_calls = [x[0] for x in expected_results]
    expected_epochs = [x[1] for x in expected_results]
    expected_batch_idx = [x[2] for x in expected_results]
    expected_num_epochs = [x[3] for x in expected_results]
    expected_num_batches_train = [x[4] for x in expected_results]
    expected_num_batches_query = [x[5] for x in expected_results]
    expected_num_batches_index = [x[6] for x in expected_results]

    assert record_callback.calls == expected_calls
    assert record_callback.epochs == expected_epochs
    assert record_callback.batch_idx == expected_batch_idx
    assert record_callback.num_epochs == expected_num_epochs
    assert record_callback.num_batches_train == expected_num_batches_train
    assert record_callback.num_batches_query == expected_num_batches_query
    assert record_callback.num_batches_index == expected_num_batches_index


def test_on_exception(exception_callback, generate_random_data):
    train_data = generate_random_data(8, 2, 2)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10, activation='relu'),
        ]
    )
    ec = exception_callback(ValueError('Test'))

    # Train
    tuner = KerasTuner(model, callbacks=[ec])

    with pytest.raises(ValueError, match='Test'):
        tuner.fit(train_data=train_data, epochs=1, batch_size=4, num_items_per_class=2)

    assert ec.calls == ['on_exception']


def test_on_keyboard_interrupt(exception_callback, generate_random_data):
    train_data = generate_random_data(8, 2, 2)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10, activation='relu'),
        ]
    )
    ec = exception_callback(KeyboardInterrupt)

    # Train
    tuner = KerasTuner(model, callbacks=[ec])

    tuner.fit(train_data=train_data, epochs=1, batch_size=4, num_items_per_class=2)

    assert ec.calls == ['on_keyboard_interrupt']
