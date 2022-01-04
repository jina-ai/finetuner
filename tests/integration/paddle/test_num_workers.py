import numpy as np
import paddle
import pytest

from finetuner.tuner.paddle import PaddleTuner


@pytest.mark.parametrize(
    'num_workers,expected_batch_time', [(0, 4), (1, 4), (2, 2), (4, 1)]
)
def test_multiple_workers(
    num_workers,
    expected_batch_time,
    generate_random_data,
    multi_workers_preprocess_fn,
    multi_workers_callback,
):
    data = generate_random_data(20, 2, 2)
    model = paddle.nn.Linear(2, 2)

    tuner = PaddleTuner(model, callbacks=[multi_workers_callback])

    tuner.fit(
        data,
        epochs=1,
        batch_size=4,
        num_items_per_class=2,
        num_workers=num_workers,
        preprocess_fn=multi_workers_preprocess_fn,
    )

    np.testing.assert_almost_equal(
        np.mean(multi_workers_callback.batch_times), expected_batch_time, decimal=1
    )
