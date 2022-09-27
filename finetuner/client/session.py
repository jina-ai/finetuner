from typing import List

from requests import Session
from requests.utils import urlparse


class _HeaderPreservingSession(Session):
    def __init__(self, trusted_domains: List[str]):
        super(_HeaderPreservingSession, self).__init__()
        self._trusted_domains = trusted_domains

    def rebuild_auth(self, prepared_request, response):
        """
        Keep headers upon redirect as long as we are on any of the
        self._trusted_domains
        """
        headers = prepared_request.headers
        url = prepared_request.url
        if 'Authorization' in headers:
            _original_parsed = urlparse(response.request.url)
            _redirect_parsed = urlparse(url)
            _original_domain = '.'.join(_original_parsed.hostname.split('.')[-2:])
            _redirect_domain = '.'.join(_redirect_parsed.hostname.split('.')[-2:])
            if (
                _original_domain != _redirect_domain
                and _original_domain not in self._trusted_domains
                and _redirect_domain not in self._trusted_domains
            ):
                del headers['Authorization']
