from typing import Any, Dict, Optional

from finetuner.client.base import BaseClient
from finetuner.constants import (
    API_VERSION,
    CONFIG,
    DELETE,
    EXPERIMENTS,
    GET,
    NAME,
    POST,
    RUNS,
    STATUS,
    DATA,
    TRAIN_DATA,
    EVAL_DATA,
    FINETUNED_MODELS_DIR,
)


class Client(BaseClient):
    def create_experiment(self, name: str, **kwargs):
        """Create a new experiment.

        :param name: The name of the experiment.
        :return: `requests.Response` object.
        """
        url = self._base_url / API_VERSION / EXPERIMENTS
        return self.handle_request(
            url=url,
            method=POST,
            json={NAME: name, **kwargs},
        )

    def get_experiment(self, name: str):
        """Get an experiment by its name.

        :param name: The name of the experiment.
        :return: `requests.Response` object.
        """
        url = self._base_url / API_VERSION / EXPERIMENTS / name
        return self.handle_request(url=url, method=GET)

    def list_experiments(self):
        """List all available experiments.

        :return: `requests.Response` object.
        """
        url = self._base_url / API_VERSION / EXPERIMENTS
        return self.handle_request(url=url, method=GET)

    def delete_experiment(self, name: str):
        """Delete an experiment given its name.

        :param name: The name of the experiment.
        :return: `requests.Response` object.
        """
        url = self._base_url / API_VERSION / EXPERIMENTS / name
        return self.handle_request(url=url, method=DELETE)

    def delete_experiments(self):
        """Delete all experiments.

        :return: `requests.Response` object.
        """
        url = self._base_url / API_VERSION / EXPERIMENTS
        return self.handle_request(url=url, method=DELETE)

    def create_run(
        self, experiment_name: str, run_name: str, config: Dict[str, Any], **kwargs
    ):
        """Create a run inside a given experiment.

        For optional parameters please visit our documentation (link).
        :param experiment_name: The name of the experiment.
        :param run_name: The name of the run.
        :param config: Configuration for the run.
        :return: `requests.Response` object.
        """
        self._push_data_to_hubble(data=config.get(DATA))
        url = self._base_url / API_VERSION / EXPERIMENTS / experiment_name / RUNS
        return self.handle_request(
            url=url,
            method=POST,
            json={NAME: run_name, CONFIG: config, **kwargs},
        )

    def get_run(self, experiment_name: str, run_name: str):
        """Get a run by its name and experiment.

        :param experiment_name: The name of the experiment.
        :param run_name: The name of the run.
        :return: `requests.Response` object.
        """
        url = (
            self._base_url
            / API_VERSION
            / EXPERIMENTS
            / experiment_name
            / RUNS
            / run_name
        )
        return self.handle_request(url=url, method=GET)

    def list_runs(self, experiment_name: Optional[str] = None):
        """List all created runs inside a given experiment.

        If no experiment is specified, list runs for all available experiments.
        :param experiment_name: The name of the experiment.
        :return: `requests.Response` object.
        """
        if not experiment_name:
            target_experiments = [
                experiment[NAME] for experiment in self.list_experiments().json()
            ]
        else:
            target_experiments = [experiment_name]
        response = []
        for experiment_name in target_experiments:
            url = self._base_url / API_VERSION / EXPERIMENTS / experiment_name / RUNS
            response.append(self.handle_request(url=url, method=GET))
        return response

    def delete_run(self, experiment_name: str, run_name: str):
        """Delete a run by its name and experiment.

        :param experiment_name: The name of the experiment.
        :param run_name: The name of the run.
        :return: `requests.Response` object.
        """
        url = (
            self._base_url
            / API_VERSION
            / EXPERIMENTS
            / experiment_name
            / RUNS
            / run_name
        )
        return self.handle_request(url=url, method=DELETE)

    def delete_runs(self, experiment_name: str):
        """Delete all runs inside a given experiment.

        :param experiment_name: The name of the experiment.
        :return: `requests.Response` object.
        """
        url = self._base_url / API_VERSION / EXPERIMENTS / experiment_name / RUNS
        return self.handle_request(url=url, method=DELETE)

    def get_run_status(self, experiment_name: str, run_name: str):
        """Get a run status by its name and experiment.

        :param experiment_name: The name of the experiment.
        :param run_name: The name of the run.
        :return: `requests.Response` object.
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
        return self.handle_request(url=url, method=GET)

    def _push_data_to_hubble(self, data: dict[str, Any]):
        """Push user's data to Hubble.

        :param data: A dictionary containing paths to train and evaluation data.
        """
        if train_path := data.get(TRAIN_DATA):
            self._hubble_client.upload_artifact(path=train_path)
        else:
            # Raise an exception. We can refactor this later? in 'handle error messages' PR.
            pass
        if eval_path := data.get(EVAL_DATA):
            self.hubble_client.upload_artifact(path=eval_path)

    def download_model(self, artifact_id: str, path: str = FINETUNED_MODELS_DIR):
        """Download finetuned model from Hubble by its ID.

        :param artifact_id: ID of the model.
        :param path: Directory where the model will be stored.
        """
        self._hubble_client.download_artifact(id=artifact_id, path=path)
