from typing import List, Optional

from finetuner.client.base import _BaseClient
from finetuner.constants import (
    API_VERSION,
    CONFIG,
    CPUS,
    DELETE,
    DESCRIPTION,
    DEVICE,
    EXPERIMENTS,
    GET,
    GPUS,
    LOGS,
    NAME,
    POST,
    RUNS,
    STATUS,
)


class FinetunerV1Client(_BaseClient):
    """
    The Finetuner v1 API client.
    """

    """ Experiment API """

    def create_experiment(self, name: str, description: Optional[str] = '') -> dict:
        """Create a new experiment.

        :param name: The name of the experiment.
        :param description: Optional description of the experiment.
        :return: Created experiment.
        """
        url = self._base_url / API_VERSION / EXPERIMENTS
        return self._handle_request(
            url=url, method=POST, json_data={NAME: name, DESCRIPTION: description}
        )

    def get_experiment(self, name: str) -> dict:
        """Get an experiment by its name.

        :param name: The name of the experiment.
        :return: Requested experiment.
        """
        url = self._base_url / API_VERSION / EXPERIMENTS / name
        return self._handle_request(url=url, method=GET)

    def list_experiments(self) -> List[dict]:
        """List all available experiments.

        :return: List of all experiments.
        """
        url = self._base_url / API_VERSION / EXPERIMENTS
        return self._handle_request(url=url, method=GET)

    def delete_experiment(self, name: str) -> dict:
        """Delete an experiment given its name.

        :param name: The name of the experiment.
        :return: Experiment to be deleted.
        """
        url = self._base_url / API_VERSION / EXPERIMENTS / name
        return self._handle_request(url=url, method=DELETE)

    def delete_experiments(self) -> List[dict]:
        """Delete all experiments.

        :return: Experiments to be deleted.
        """
        url = self._base_url / API_VERSION / EXPERIMENTS
        return self._handle_request(url=url, method=DELETE)

    """ Run API """

    def get_run(self, experiment_name: str, run_name: str) -> dict:
        """Get a run by its name and experiment.

        :param experiment_name: The name of the experiment.
        :param run_name: The name of the run.
        :return: Requested run.
        """
        url = (
            self._base_url
            / API_VERSION
            / EXPERIMENTS
            / experiment_name
            / RUNS
            / run_name
        )
        return self._handle_request(url=url, method=GET)

    def list_runs(self, experiment_name: Optional[str] = None) -> List[dict]:
        """List all created runs inside a given experiment.

        If no experiment is specified, list runs for all available experiments.
        :param experiment_name: The name of the experiment.
        :return: List of all runs.
        """
        if not experiment_name:
            target_experiments = [
                experiment[NAME] for experiment in self.list_experiments()
            ]
        else:
            target_experiments = [experiment_name]
        response = []
        for experiment_name in target_experiments:
            url = self._base_url / API_VERSION / EXPERIMENTS / experiment_name / RUNS
            response.extend(self._handle_request(url=url, method=GET))
        return response

    def delete_run(self, experiment_name: str, run_name: str) -> dict:
        """Delete a run by its name and experiment.

        :param experiment_name: The name of the experiment.
        :param run_name: The name of the run.
        :return: Deleted run.
        """
        url = (
            self._base_url
            / API_VERSION
            / EXPERIMENTS
            / experiment_name
            / RUNS
            / run_name
        )
        return self._handle_request(url=url, method=DELETE)

    def delete_runs(self, experiment_name: str) -> List[dict]:
        """Delete all runs inside a given experiment.

        :param experiment_name: The name of the experiment.
        :return: List of all deleted runs.
        """
        url = self._base_url / API_VERSION / EXPERIMENTS / experiment_name / RUNS
        return self._handle_request(url=url, method=DELETE)

    def get_run_status(self, experiment_name: str, run_name: str) -> dict:
        """Get a run status by its name and experiment.

        :param experiment_name: The name of the experiment.
        :param run_name: The name of the run.
        :return: Run status.
        """
        url = (
            self._base_url
            / API_VERSION
            / EXPERIMENTS
            / experiment_name
            / RUNS
            / run_name
            / STATUS
        )
        return self._handle_request(url=url, method=GET)

    def get_run_logs(self, experiment_name: str, run_name: str) -> str:
        """Get a run logs by its name and experiment.

        :param experiment_name: The name of the experiment.
        :param run_name: The name of the run.
        :return: Run logs.
        """
        url = (
            self._base_url
            / API_VERSION
            / EXPERIMENTS
            / experiment_name
            / RUNS
            / run_name
            / LOGS
        )
        return self._handle_request(url=url, method=GET)

    def create_run(
        self,
        experiment_name: str,
        run_name: str,
        run_config: dict,
        device: str,
        cpus: int,
        gpus: int,
    ) -> dict:
        """Create a run inside a given experiment.

        For optional parameters please visit our documentation (link).
        :param experiment_name: The name of the experiment.
        :param run_name: The name of the run.
        :param run_config: The run configuration.
        :param device: The device to use, either `cpu` or `gpu`.
        :param cpus: The number of CPUs to use.
        :param gpus: The number of GPUs to use.
        :return: Created run.
        """
        url = self._base_url / API_VERSION / EXPERIMENTS / experiment_name / RUNS
        return self._handle_request(
            url=url,
            method=POST,
            json_data={
                NAME: run_name,
                CONFIG: run_config,
                DEVICE: device,
                CPUS: cpus,
                GPUS: gpus,
            },
        )
