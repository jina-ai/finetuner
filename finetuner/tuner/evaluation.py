import numpy as np
from jina import Document, DocumentArray
from .. import __default_tag_key__


def prepate_eval_docs(data, limit=100, sample_size=100, seed=42):
    sampled_docs = data.sample(sample_size, seed)
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
        print(d.embedding)
        DocumentArray([d]).match(data, limit=limit)
        to_be_scored_docs.append(d)
    return to_be_scored_docs


def get_ndcg(data, to_be_scored_docs, limit=100):
    ndcg = 0
    for doc in to_be_scored_docs:
        dcg = 0
        positive_ids = doc.tags['positive_ids']
        pos_counter = 0
        for position, match in enumerate(doc.matches):
            if match.id in positive_ids:
                dcg += 1 / np.log(position + 2)
                pos_counter += 1

        idcg = max(_get_idcg(pos_counter), 1e-10)
        ndcg += dcg / idcg
    return ndcg


def _get_idcg(n):
    return sum(1 / np.log(position + 2) for position in range(n))


def get_map(data):
    pass


'''

usual data structure:
- doc
    - match (labeled data)
    - match (labeled data)


- compute distances and order results in:

NDCG @ X data structure:
- doc
    - match (actual matches according to model distance)
    - match (actual matches according to model distance)
     ....


sum(doc of docs) ...

100k items in shop
100
100k * 100

10 million clicks:

9M => train data
1M => eval data
100k Documents with 1M labels
NDCG only for 1k Documents: remove labels(matches) from 99k Documents

Doc1
    - 50 positive
    - 50 negative
    - 1000 unknown
'''
