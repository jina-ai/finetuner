import json
import os
from typing import Optional

import hubble
import requests
from path import Path

from finetuner.constants import (
    AUTHORIZATION,
    CHARSET,
    DATA,
    HOST,
    HUBBLE_USER_ID,
    TOKEN_PREFIX,
    UTF_8,
)


class BaseClient(object):
    """Base Finetuner API client."""

    def __init__(self):
        self._base_url = Path(os.environ.get(HOST))
        self._session = self._get_client_session()
        self._hubble_client = hubble.Client(max_retries=None, timeout=10, jsonify=True)
        self._hubble_user_id = self._get_hubble_user_id()

    def _get_hubble_user_id(self):
        user_info = json.loads(self._hubble_client.get_user_info())
        if user_info['code'] >= 400:
            # will implement error-handling later
            pass
        hubble_user_id = user_info[DATA][HUBBLE_USER_ID]
        return hubble_user_id

    @staticmethod
    def _get_client_session() -> requests.Session:
        session = requests.Session()
        api_token = TOKEN_PREFIX + str(hubble.Auth.get_auth_token())
        session.headers.update({CHARSET: UTF_8, AUTHORIZATION: api_token})
        return session

    def _handle_request(
        self,
        url: str,
        method: str,
        params: Optional[dict] = None,
        json_data: Optional[dict] = None,
    ) -> requests.Response:
        """The base request handler.

        :param url: The url of the request.
        :param method: The request type (GET, POST or DELETE).
        :param params: Optional parameters for the request.
        :param json_data: Optional data payloads to be sent along with the request.
        :return: `requests.Response` object
        """
        response = self._session.request(
            url=url, method=method, json=json_data, params=params
        )
        return response
