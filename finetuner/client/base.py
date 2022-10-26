import os
from typing import List, Optional, Union

import requests

import hubble
from finetuner.client.session import _HeaderPreservingSession
from finetuner.constants import (
    AUTHORIZATION,
    CHARSET,
    DATA,
    HOST,
    HUBBLE_USER_ID,
    TEXT,
    TOKEN_PREFIX,
    UTF_8,
)
from finetuner.excepts import FinetunerServerError


class _BaseClient:
    """
    Base Finetuner API client.
    """

    def __init__(self):
        self._base_url = os.environ.get(HOST)
        self._session = self._get_client_session()
        self.hubble_client = hubble.Client(max_retries=None, jsonify=True)
        self.hubble_user_id = self._get_hubble_user_id()

    def _get_hubble_user_id(self):
        user_info = self.hubble_client.get_user_info()
        if user_info['code'] >= 400:
            # will implement error-handling later
            pass
        hubble_user_id = user_info[DATA][HUBBLE_USER_ID]
        return hubble_user_id

    @staticmethod
    def _get_client_session() -> _HeaderPreservingSession:
        session = _HeaderPreservingSession(trusted_domains=[])
        api_token = TOKEN_PREFIX + str(hubble.Auth.get_auth_token())
        session.headers.update({CHARSET: UTF_8, AUTHORIZATION: api_token})
        return session

    @staticmethod
    def _construct_url(*args) -> str:
        return '/'.join(args)

    def _handle_request(
        self,
        url: str,
        method: str,
        params: Optional[dict] = None,
        json_data: Optional[dict] = None,
        stream: bool = False,
    ) -> Union[dict, List[dict], str, requests.Response]:
        """The base request handler.

        :param url: The url of the request.
        :param method: The request type (GET, POST or DELETE).
        :param params: Optional parameters for the request.
        :param json_data: Optional data payloads to be sent along with the request.
        :param stream: If the request is a streaming request set to True.
        :return: Response to the request.
        """
        response = self._session.request(
            url=url,
            method=method,
            json=json_data,
            params=params,
            allow_redirects=True,
            stream=stream,
        )
        if not response.ok:
            raise FinetunerServerError(
                message=response.reason,
                code=response.status_code,
                details=response.json()['detail'],
            )
        if stream:
            return response
        else:
            if TEXT in response.headers['content-type']:
                return response.text
            return response.json()
