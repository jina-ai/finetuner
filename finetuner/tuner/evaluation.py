import numpy as np
from jina import Document, DocumentArray
from .. import __default_tag_key__


def prepare_eval_docs(docs, catalog, limit=10, sample_size=100, seed=42):
    sampled_docs = docs.sample(min(sample_size, len(docs)), seed)
    to_be_scored_docs = DocumentArray()
    for doc in sampled_docs:
        d = Document(
            id=doc.id,
            embedding=doc.embedding,
            tags={
                'positive_ids': [
                    m.id
                    for m in doc.matches
                    if m.tags[__default_tag_key__]['label'] > 0
                ]
            },
        )
        to_be_scored_docs.append(d)
    to_be_scored_docs.match(catalog, limit=limit)
    return to_be_scored_docs


def get_hits_at_n(to_be_scored_docs, n=-1):
    hits = 0
    for doc in to_be_scored_docs:
        positive_ids = doc.tags['positive_ids']
        for match in doc.matches[:n]:
            if match.id in positive_ids:
                hits += 1
    return hits


def get_ndcg_at_n(to_be_scored_docs, n=-1):
    ndcg = 0
    for doc in to_be_scored_docs:
        dcg = 0
        positive_ids = doc.tags['positive_ids']
        first_n = doc.matches[:n]
        for position, match in enumerate(first_n):
            if match.id in positive_ids:
                dcg += 1 / np.log(position + 2)

        max_positives = min(len(positive_ids), len(first_n))
        idcg = max(_get_idcg(max_positives), 1e-10)
        ndcg += dcg / idcg
    return ndcg


def _get_idcg(n):
    return sum(1 / np.log(position + 2) for position in range(n))
