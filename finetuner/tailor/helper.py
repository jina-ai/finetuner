from typing import Sequence, Dict, List, Any

CandidateLayerInfo = List[Dict[str, Any]]


def _is_list_int(tp) -> bool:
    """Return True if the input is a list of integers."""
    return tp and isinstance(tp, Sequence) and all(isinstance(p, int) for p in tp)
