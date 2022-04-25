import os
from typing import Optional

import requests
from path import Path

from finetuner.constants import HOST


class BaseClient(object):
    """Base Finetuner API client.

    :param user_id: Unique identifier of a user.
    """

    def __init__(self, user_id):
        self._user_id = user_id
        self._session = requests.session()
        self._base_url = Path(os.environ.get(HOST))
        self._session.headers.update({'Accept-Charset': 'utf-8'})

    def handle_request(
        self,
        url: str,
        method: str,
        params: Optional[dict] = None,
        json: Optional[dict] = None,
    ):
        """The basis request handler.

        :param url: The url of the request.
        :param method: The request type (GET, POST or DELETE).
        :param params: Optional parameters for the request.
        :param json: Optional data payloads to be send along with the request.
        :return: `requests.Response` object
        """
        response = self._session.request(
            url=url, method=method, json=json, params=params
        )
        return response
