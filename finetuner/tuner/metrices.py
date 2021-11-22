import numpy as np


def hits_at_n(doc, n=-1):
    hits = 0
    positive_ids = doc.tags['positive_ids']
    for match in doc.matches[:n]:
        if match.id in positive_ids:
            hits += 1
    return hits


def ndcg_at_n(doc, n=-1):
    dcg = 0
    positive_ids = doc.tags['positive_ids']
    first_n = doc.matches[:n]
    for position, match in enumerate(first_n):
        if match.id in positive_ids:
            dcg += 1 / np.log(position + 2)

    max_positives = min(len(positive_ids), len(first_n))
    idcg = max(_idcg_at_n(max_positives), 1e-10)
    return dcg / idcg


def _idcg_at_n(n):
    return sum(1 / np.log(position + 2) for position in range(n))
