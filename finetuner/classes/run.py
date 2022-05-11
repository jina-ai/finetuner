from typing import Optional

from finetuner.client.client import Client
from finetuner.constants import FINETUNED_MODELS_DIR, FINISHED


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
        client: Client,
        name: str,
        experiment_name: str,
        config: dict,
        created_at: str,
        description: Optional[str] = "",
    ):
        self._client = client
        self._name = name
        self._experiment_name = experiment_name
        self._config = config
        self._created_at = created_at
        self._description = description

    def status(self) -> str:
        """Run status.

        :returns: A string corresponding to four possible states of the run: CREATED, STARTED, FINISHED AND FAILED.
        """
        return self._client.get_run_status(
            experiment_name=self.experiment_name, run_name=self.name
        )

    def save_model(self, path: str = FINETUNED_MODELS_DIR):
        """Save model(s) if the run is finished.

        :param path: Directory where the model(s) will be stored.
        :returns: A list of str object(s) that indicate the download path.
        """
        if self.status() == FINISHED:
            download_path = self._client.download_model(
                experiment_name=self.experiment_name, run_name=self.name, path=path
            )
        else:
            # tell users that the run hasn't finished yet, thus they can't download the model.
            pass
        return download_path
