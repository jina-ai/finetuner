from typing import Optional


class BaseError(Exception):
    code = -1
    message = "An unknown error occurred"

    def __init__(
        self,
        message: Optional[str] = None,
        code: Optional[int] = None,
    ):
        if message:
            self.message = message
        if code:
            self.code = code

    def __str__(self):
        if self.code:
            return f'{self.code}: {self.message}'
        return self.message
