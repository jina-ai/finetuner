import time
from typing import Iterator

from finetuner.client import FinetunerV1Client
from finetuner.constants import (
    ARTIFACT_ID,
    ARTIFACTS_DIR,
    CREATED,
    FAILED,
    STARTED,
    STATUS,
)
from finetuner.exception import RunFailedError, RunInProgressError, RunPreparingError
from finetuner.hubble import download_artifact


class Run:
    """Class for a run.

    :param client: Client object for sending api requests.
    :param name: Name of the run.
    :param experiment_name: Name of the experiment.
    :param config: Configuration for the run.
    :param created_at: Creation time of the run.
    :param description: Optional description of the run.
    """

    def __init__(
        self,
        client: FinetunerV1Client,
        name: str,
        experiment_name: str,
        config: dict,
        created_at: str,
        description: str = '',
    ):
        self._client = client
        self._name = name
        self._experiment_name = experiment_name
        self._config = config
        self._created_at = created_at
        self._description = description
        self._run = self._get_run()

    @property
    def name(self) -> str:
        return self._name

    @property
    def config(self) -> dict:
        return self._config

    def _get_run(self) -> dict:
        """Get Run object as dict."""
        return self._client.get_run(
            experiment_name=self._experiment_name, run_name=self._name
        )

    def status(self) -> dict:
        """Run status.

        :returns: A string representing the run status.
        """
        return self._client.get_run_status(
            experiment_name=self._experiment_name, run_name=self._name
        )

    def logs(self) -> str:
        """Check the run logs.

        :returns: A string dump of the run logs.
        """
        self._check_run_status_started()
        return self._client.get_run_logs(
            experiment_name=self._experiment_name, run_name=self._name
        )

    def stream_logs(self, interval: int = 5) -> Iterator[str]:
        """Stream the run logs.

        :param interval: The time interval to sync the status of finetuner `Run`.
        :yield: An iterators keep stream the logs from server.
        """
        while True:
            time.sleep(interval)
            status = self.status()[STATUS]
            if status == CREATED:
                msg = (
                    f'Preparing to run, logs will be ready to pull when '
                    f'`status` is `STARTED`. Current status is {status}'
                )
                print(msg)
            else:
                break
        return self._client.stream_run_logs(
            experiment_name=self._experiment_name, run_name=self._name
        )

    def _check_run_status_finished(self):
        status = self.status()[STATUS]
        if status in [CREATED, STARTED]:
            raise RunInProgressError(
                'The run needs to be finished in order to save the artifact.'
            )
        if status == FAILED:
            raise RunFailedError(
                'The run failed, please check the `logs` for detailed information.'
            )

    def _check_run_status_started(self):
        status = self.status()[STATUS]
        if status == CREATED:
            raise RunPreparingError(
                'Preparing to run, logs will be ready to pull when '
                '`status` is `STARTED`.'
            )

    def save_artifact(self, directory: str = ARTIFACTS_DIR) -> str:
        """Save artifact if the run is finished.

        :param directory: Directory where the artifact will be stored.
        :returns: A string object that indicates the download path.
        """
        self._check_run_status_finished()
        return download_artifact(
            client=self._client,
            artifact_id=self._run[ARTIFACT_ID],
            run_name=self._name,
            directory=directory,
        )

    @property
    def artifact_id(self):
        """Get artifact id from the run.

        An artifact in finetuner contains fine-tuned model and its metadata.
        Such as preprocessing function, collate function. This id could be useful
        if you want to directly pull the artifact from the cloud storage, such as
        using `FinetunerExecutor`.

        :return: Artifact id as string object.
        """
        self._check_run_status_finished()
        return self._run[ARTIFACT_ID]
