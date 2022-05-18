from finetuner.constants import CREATED, FAILED, FINISHED, STARTED
from finetuner.run import Run


def test_run_obj(test_finetuner):
    test_config = {'type': 'test'}
    run = Run(
        client=test_finetuner._client,
        name='run name',
        experiment_name='exp name',
        config=test_config,
        created_at='some time',
        description='description',
    )

    assert run.get_name() == 'run name'
    assert run.status() in [CREATED, STARTED, FINISHED, FAILED]
    assert run.get_config() == test_config
