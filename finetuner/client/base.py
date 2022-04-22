from typing import Optional

import requests
from path import Path


class BaseClient(object):
    def __init__(self):
        self._session = requests.session()
        self._base_url = Path("http://18.192.5.194")
        self._session.headers.update({'Accept-Charset': 'utf-8'})

    def handle_request(
        self,
        url: str,
        method: str,
        params: Optional[dict] = None,
        data: Optional[dict] = None,
    ):
        response = self._session.request(
            url=url, method=method, json=data, params=params
        )

        return response
