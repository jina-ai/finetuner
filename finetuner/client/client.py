from finetuner.client.base import BaseClient
from finetuner.client.endpoints import Endpoints
from finetuner.constants import DELETE, GET, NAME, POST, USER_ID


class Client(BaseClient):
    def create_experiment(self, name: str):
        """Create a new experiment.

        :param name: The name of the experiment
        :return: `requests.Response` object
        """
        url = self._base_url / Endpoints.experiments
        return self.handle_request(
            url=url,
            method=POST,
            data={NAME: name},
            params={USER_ID: self._user_id},
        )

    def get_experiment(self, name: str):
        """Get an experiment by its name.

        :param name: The name of the experiment
        :return: `requests.Response` object
        """
        url = self._base_url / Endpoints.experiments / name
        return self.handle_request(url=url, method=GET, params={USER_ID: self._user_id})

    def list_experiments(self):
        """List all available experiments.

        :return: `requests.Response` object
        """
        url = self._base_url / Endpoints.experiments
        return self.handle_request(url=url, method=GET, params={USER_ID: self._user_id})

    def delete_experiment(self, name: str):
        """Delete an experiment given its name.

        :param name: The name of the experiment
        :return: `requests.Response` object
        """
        url = self._base_url / Endpoints.experiments / name
        return self.handle_request(
            url=url, method=DELETE, params={USER_ID: self._user_id}
        )

    def delete_experiments(self):
        """Delete all experiments.

        :return: `requests.Response` object
        """
        url = self._base_url / Endpoints.experiments
        return self.handle_request(
            url=url, method=DELETE, params={USER_ID: self._user_id}
        )
