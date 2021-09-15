from typing import Sequence


def is_list_int(tp):
    return tp and isinstance(tp, Sequence) and all(isinstance(p, int) for p in tp)
