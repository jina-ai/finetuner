import random
import string


def create_random_name(prefix='experiment', length=6):
    return f'{prefix}-' + ''.join(
        random.choices(string.ascii_uppercase + string.digits, k=length)
    )
