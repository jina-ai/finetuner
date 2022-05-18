import docarray
import pytest

from finetuner.constants import EVAL_DATA, TRAIN_DATA
from finetuner.hubble import push_data_to_hubble


@pytest.mark.parametrize(
    'data, data_type',
    [
        [docarray.DocumentArray.empty(1), TRAIN_DATA],
        [docarray.DocumentArray.empty(1), EVAL_DATA],
        ['train data', TRAIN_DATA],
        ['eval data', EVAL_DATA],
    ],
)
def test_push_data_to_hubble(
    client_mocker, data, data_type, experiment_name='exp', run_name='run'
):
    if isinstance(data, docarray.DocumentArray):
        expected_name = '-'.join(
            [client_mocker.hubble_user_id, experiment_name, run_name, data_type]
        )
    else:
        expected_name = data

    da_name = push_data_to_hubble(
        client=client_mocker,
        data=data,
        data_type=data_type,
        experiment_name=experiment_name,
        run_name=run_name,
    )

    assert da_name == expected_name
