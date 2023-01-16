import time
from typing import Iterator

from finetuner.client import FinetunerV1Client
from finetuner.console import console
from finetuner.constants import (
    ARTIFACT_ID,
    ARTIFACTS_DIR,
    CREATED,
    FAILED,
    STARTED,
    STATUS,
)
from finetuner.excepts import RunFailedError, RunInProgressError, RunPreparingError
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
        """Get the name of the :class:`Run`."""
        return self._name

    @property
    def config(self) -> dict:
        """Get the config of the :class:`Run`."""
        return self._config

    def _get_run(self) -> dict:
        """Get Run object as dict."""
        return self._client.get_run(
            experiment_name=self._experiment_name, run_name=self._name
        )

    def status(self) -> dict:
        """Get :class:`Run` status.

        :returns: A dict representing the :class:`Run` status.
        """
        return self._client.get_run_status(
            experiment_name=self._experiment_name, run_name=self._name
        )

    def logs(self) -> str:
        """Check the :class:`Run` logs.

        :returns: A string dump of the run logs.
        """
        self._check_run_status_started()
        return self._client.get_run_logs(
            experiment_name=self._experiment_name, run_name=self._name
        )

    def stream_logs(self, interval: int = 5) -> Iterator[str]:
        """Stream the :class:`Run` logs lively.

        :param interval: The time interval to sync the status of finetuner `Run`.
        :yield: An iterators keep stream the logs from server.
        """
        status = self.status()[STATUS]
        msg_template = (
            'Preparing to run, logs will be ready to pull when '
            '`status` is `STARTED`. Current status is `%s`'
        )
        with console.status(msg_template % status, spinner="dots") as rich_status:
            while status == CREATED:
                time.sleep(interval)
                status = self.status()[STATUS]
                rich_status.update(msg_template % status)

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
        """Save artifact if the :class:`Run` is finished.

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
        """Get artifact id of the :class:`Run`.

        An artifact in finetuner contains fine-tuned model and its metadata.
        Such as preprocessing function, collate function. This id could be useful
        if you want to directly pull the artifact from the cloud storage, such as
        using `FinetunerExecutor`.

        :return: Artifact id as string object.
        """
        self._check_run_status_finished()
        return self._run[ARTIFACT_ID]
