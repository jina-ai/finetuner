import pytest
import docarray
from finetuner.constants import TRAIN_DATA, EVAL_DATA


@pytest.mark.parametrize(
    'train_data, eval_data',
    [
        [docarray.DocumentArray(), docarray.DocumentArray()],
        [docarray.DocumentArray(), None],
        ['train data', 'eval data'],
        ['train data', None],
    ],
)
def test_push_data_to_hubble(
    test_client, train_data, eval_data, experiment_name='exp', run_name='run'
):
    if isinstance(train_data, docarray.DocumentArray):
        expected_train_name = '-'.join(
            [test_client._hubble_user_id, experiment_name, run_name, TRAIN_DATA]
        )
    else:
        expected_train_name = train_data
    if isinstance(eval_data, docarray.DocumentArray):
        expected_eval_name = '-'.join(
            [test_client._hubble_user_id, experiment_name, run_name, EVAL_DATA]
        )
    else:
        expected_eval_name = eval_data

    train_da_name, eval_da_name = test_client._push_data_to_hubble(
        train_data=train_data,
        eval_data=eval_data,
        experiment_name=experiment_name,
        run_name=run_name,
    )

    assert train_da_name == expected_train_name
    assert eval_da_name == expected_eval_name
