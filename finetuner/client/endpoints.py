from dataclasses import dataclass


@dataclass(frozen=True)
class Endpoints(object):
    experiments: str = "api/v1/experiments/"
