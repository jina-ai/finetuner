from ..constants import DELETE, GET, NAME, POST, USER_ID
from .base import BaseClient
from .endpoints import Endpoints


class Client(BaseClient):
    def __init__(self, user_id: str):
        super(Client, self).__init__()
        self._user_id = user_id

    def create_experiment(self, experiment_name: str):
        url = self._base_url / Endpoints.experiments
        data_dict = {NAME: experiment_name}
        return self.handle_request(
            url=url, method=POST, data=data_dict, params={USER_ID: self._user_id}
        )

    def get_experiment(self, experiment_name: str):
        url = self._base_url / Endpoints.experiments / experiment_name
        return self.handle_request(url=url, method=GET, params={USER_ID: self._user_id})

    def get_experiments(self):
        url = self._base_url / Endpoints.experiments
        return self.handle_request(url=url, method=GET, params={USER_ID: self._user_id})

    def delete_experiment(self, experiment_name: str):
        url = self._base_url / Endpoints.experiments / experiment_name
        return self.handle_request(
            url=url, method=DELETE, params={USER_ID: self._user_id}
        )

    def delete_experiments(self):
        url = self._base_url / Endpoints.experiments
        return self.handle_request(
            url=url, method=DELETE, params={USER_ID: self._user_id}
        )
