from typing import List, Optional, Union

from docarray import DocumentArray

from ..client import FinetunerV1Client
from ..constants import CONFIG, CREATED_AT, DESCRIPTION, NAME
from .run import Run


class Experiment:
    """Class for an experiment.

    :param client: Client object for sending api requests.
    :param name: Name of the experiment.
    :param status: Status of the experiment.
    :param created_at: Creation time of the experiment.
    :param description: Optional description of the experiment.
    """

    def __init__(
        self,
        client: FinetunerV1Client,
        name: str,
        status: str,
        created_at: str,
        description: Optional[str] = "",
    ):
        self._client = client
        self._name = name
        self._status = status
        self._created_at = created_at
        self._description = description

    def get_run(self, name: str) -> Run:
        """Get a run by its name.

        :param name: Name of the run.
        :returns: A `Run` object.
        """
        run_info = self._client.get_run(experiment_name=self._name, run_name=name)
        run = Run(name=run_info[NAME], experiment_name=self._name, client=self._client)
        return run

    def list_runs(self) -> List[Run]:
        """List every run inside the experiment.

        :returns: List of `Run` objects.
        """
        run_infos = self._client.list_runs(experiment_name=self._name)
        runs = [
            Run(name=run_info[NAME], experiment_name=self._name, client=self._client)
            for run_info in run_infos
        ]
        return runs

    def delete_run(self, name: str):
        """Delete a run by its name.

        :param name: Name of the run.
        """
        self._client.delete_run(experiment_name=self._name, run_name=name)

    def delete_runs(self):
        """Delete every run inside the experiment."""
        self._client.delete_runs(experiment_name=self._name)

    def create_run(
        self,
        run_name: str,
        model: str,
        train_data: Union[DocumentArray, str],
        **kwargs,
    ) -> Run:
        """Create a run inside the experiment.

        :param run_name: Name of the run.
        :param model: Name of the model to be fine-tuned.
        :param train_data: Either a `DocumentArray` for training data or a
                           name of the `DocumentArray` that is pushed on Hubble.
        :returns: A `Run` object.
        """
        run_info = self._client.create_run(
            run_name=run_name,
            experiment_name=self._name,
            train_data=train_data,
            model=model,
            **kwargs,
        )
        run = Run(
            client=self._client,
            name=run_info[NAME],
            experiment_name=self._name,
            config=run_info[CONFIG],
            created_at=run_info[CREATED_AT],
            description=run_info[DESCRIPTION],
        )
        return run
