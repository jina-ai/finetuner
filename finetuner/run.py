from finetuner.client import FinetunerV1Client
from finetuner.constants import FINETUNED_MODEL, FINISHED, STATUS
from finetuner.hubble import download_model


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

    @property
    def name(self) -> str:
        return self._name

    @property
    def config(self) -> dict:
        return self._config

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
        return self._client.get_run_logs(
            experiment_name=self._experiment_name, run_name=self._name
        )

    def save_model(self, path: str = FINETUNED_MODEL):
        """Save model(s) if the run is finished.

        :param path: Path where the model(s) will be stored.
        :returns: A list of str object(s) that indicate the download path.
        """
        if self.status()[STATUS] != FINISHED:
            raise Exception('The run needs to be finished in order to save the model.')

        download_path = download_model(
            client=self._client,
            experiment_name=self._experiment_name,
            run_name=self._name,
            path=path,
        )
        return download_path
