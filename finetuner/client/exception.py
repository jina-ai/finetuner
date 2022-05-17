from typing import Optional


class FinetunerServerError(Exception):
    def __init__(
        self,
        message: Optional[str] = 'An unknown error occurred',
        code: Optional[int] = -1,
    ):
        if message:
            self.message = message
        if code:
            self.code = code

    def __str__(self):
        if self.code:
            return f'{self.code}: {self.message}'
        return self.message
