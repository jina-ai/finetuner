from finetuner.constants import CREATED, FAILED, FINISHED, STARTED, STATUS
from finetuner.run import Run


def test_run_obj(finetuner_mocker):
    test_config = {'type': 'test'}
    run = Run(
        client=finetuner_mocker._client,
        name='run name',
        experiment_name='exp name',
        config=test_config,
        created_at='some time',
        description='description',
    )

    assert run.name == 'run name'
    assert run.status()[STATUS] in [CREATED, STARTED, FINISHED, FAILED]
    assert run.config == test_config
