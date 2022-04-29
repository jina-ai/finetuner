import pytest
import docarray


@pytest.mark.parametrize(
    'train_data, eval_data, expected_train, expected_eval',
    [[docarray.DocumentArray(), docarray.DocumentArray(), 'da name', 'da name'],
     [docarray.DocumentArray(), None, 'da name', None],
     ['train data', 'eval data', 'train data', 'eval data'],
     ['train data', None, 'train data', None]]
)
def test_push_data_to_hubble(test_client, train_data, eval_data, expected_train, expected_eval):
    train_da_name, eval_da_name = test_client._push_data_to_hubble(train_data, eval_data)
    assert train_da_name == expected_train
    assert eval_da_name == expected_eval
