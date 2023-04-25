import docarray

from finetuner.constants import DA_PREFIX
from finetuner.hubble import push_training_data


def test_push_training_data(client_mocker, experiment_name='exp', run_name='run'):
    train_data = docarray.DocumentArray.empty(10)
    eval_data = query_data = docarray.DocumentArray.empty(5)
    index_data = None

    train_name, eval_name, query_name, index_name = push_training_data(
        experiment_name=experiment_name,
        run_name=run_name,
        train_data=train_data,
        eval_data=eval_data,
        query_data=query_data,
        index_data=index_data,
    )
    assert train_name == f'{DA_PREFIX}-{experiment_name}-{run_name}-train'
    assert eval_name == query_name == f'{DA_PREFIX}-{experiment_name}-{run_name}-eval'
    assert not index_name
