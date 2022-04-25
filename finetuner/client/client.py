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
    USER_ID,
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
            params={USER_ID: self._user_id},
        )

    def get_experiment(self, name: str):
        """Get an experiment by its name.

        :param name: The name of the experiment.
        :return: `requests.Response` object.
        """
        url = self._base_url / API_VERSION / EXPERIMENTS / name
        return self.handle_request(url=url, method=GET, params={USER_ID: self._user_id})

    def list_experiments(self, **kwargs):
        """List all available experiments.

        :return: `requests.Response` object.
        """
        url = self._base_url / API_VERSION / EXPERIMENTS
        return self.handle_request(
            url=url, method=GET, params={USER_ID: self._user_id, **kwargs}
        )

    def delete_experiment(self, name: str):
        """Delete an experiment given its name.

        :param name: The name of the experiment.
        :return: `requests.Response` object.
        """
        url = self._base_url / API_VERSION / EXPERIMENTS / name
        return self.handle_request(
            url=url, method=DELETE, params={USER_ID: self._user_id}
        )

    def delete_experiments(self):
        """Delete all experiments.

        :return: `requests.Response` object.
        """
        url = self._base_url / API_VERSION / EXPERIMENTS
        return self.handle_request(
            url=url, method=DELETE, params={USER_ID: self._user_id}
        )

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
        url = self._base_url / API_VERSION / EXPERIMENTS / experiment_name / RUNS
        return self.handle_request(
            url=url,
            method=POST,
            json={NAME: run_name, CONFIG: config, **kwargs},
            params={USER_ID: self._user_id},
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
        return self.handle_request(url=url, method=GET, params={USER_ID: self._user_id})

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
            response.append(
                self.handle_request(
                    url=url, method=GET, params={USER_ID: self._user_id}
                )
            )
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
        return self.handle_request(
            url=url, method=DELETE, params={USER_ID: self._user_id}
        )

    def delete_runs(self, experiment_name: str):
        """Delete all runs inside a given experiment.

        :param experiment_name: The name of the experiment.
        :return: `requests.Response` object.
        """
        url = self._base_url / API_VERSION / EXPERIMENTS / experiment_name / RUNS
        return self.handle_request(
            url=url, method=DELETE, params={USER_ID: self._user_id}
        )

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
        return self.handle_request(url=url, method=GET, params={USER_ID: self._user_id})
