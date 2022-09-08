class FinetunerServerError(Exception):
    def __init__(
        self,
        message: str = 'An unknown error occurred',
        details: str = '',
        code: int = -1,
    ):
        self.details = details
        self.message = message
        self.code = code

    def __str__(self):
        return f'{self.message} ({self.code}): {self.details}'


class RunInProgressError(Exception):
    ...


class RunPreparingError(Exception):
    ...


class RunFailedError(Exception):
    ...
