import os
from typing import Optional

import hubble
import requests
from path import Path

from finetuner.constants import HOST, AUTHORIZATION, CHARSET, UTF_8, TOKEN_PREFIX


class BaseClient(object):
    """Base Finetuner API client."""

    def __init__(self):
        self._base_url = Path(os.environ.get(HOST))
        self._session = self._get_client_session()
        self._hubble_client = hubble.Client(max_retries=None, timeout=10, jsonify=True)

    @staticmethod
    def _get_client_session():
        session = requests.session()
        api_token = TOKEN_PREFIX + str(hubble.Auth.get_auth_token())
        session.headers.update({CHARSET: UTF_8, AUTHORIZATION: api_token})
        return session

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
