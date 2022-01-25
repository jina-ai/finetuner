# Evaluation

Information retrieval metrics like NDCG and Reciprocal Rank can be calculated using the included 
{class}`~finetuner.tuner.evaluation.Evaluator` component. The evaluator can be used
[standalone](#using-the-evaluator)
or integrated in the training loop via the [evaluation callback](#using-the-evaluation-callback).


## Using the evaluator

The evaluator can be used standalone in order to compute evaluation metrics on a sequence
of documents:
```python
from finetuner.tuner.evaluation import Evaluator
from docarray import DocumentArray

query_data = DocumentArray(...)
index_data = DocumentArray(...)
embed_model = ...

evaluator = Evaluator(query_data, index_data, embed_model)

metrics = evaluator.evaluate()
```

The `query_data` (or eval data) are the documents that will be evaluated. They can be in both the
{term}`class dataset` or the {term}`session dataset` format. They should contain ground truths, in the form of
matches (`doc.matches`) when using session format and in the form of labels when using class format.

If an embedding model is given, both query and index docs are embedded, otherwise they are assumed to carry
representations.

The `index_data` (or catalog) is an optional argument that defines the dataset against which the
query docs are matched. If not provided, query docs are matched against themselves.

The `evaluate()` method returns the computed metrics as a dictionary, mapping metric names to values.
By default, the following metrics are computed:

- Precision
- Recall
- F1 Score
- Hit Score
- R precision
- Average Precision
- Reciprocal Rank
- DCG
- NDCG

More details on these Information Retrieval metrics can be found
[here](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)).

Let's consider an example. First, let's build a model that receives a NumPy array with a single element as input, and
outputs a vector of size 10, with all the elements set to the input scalar:
```python
import torch

class EmbeddingModel(torch.nn.Module):
    
    @staticmethod
    def forward(inputs):
        """
        input_shape: (bs, 1)
        output_shape: (bs, 10)
        """
        return inputs.repeat(1, 10)
```

Now let's create some example data. We will divide into 2 sets, the evaluation data and the index data. The
evaluation data are docs with ids from 0 to 9. The index data are docs with ids from 10 to 19. As content, for
each doc we use `doc.tensor = np.array([doc.id % 10])`. Same content yields the same embedding, so for each
query doc we set as target match, the corresponding doc from the index data, that has the same content. For
class data, we assign docs with the same content to the same class label:

````{tab} Session data
```python
import numpy as np
from docarray import Document,  DocumentArray

query_data = DocumentArray()
for i in range(10):
    doc = Document(
        id=str(i),
        tensor=np.array([i]),
        matches=[Document(id=str(10 + i))],
    )
    query_data.append(doc)

index_data = DocumentArray()
for i in range(10):
    doc = Document(
        id=str(i + 10),
        tensor=np.array([i]),
    )
    index_data.append(doc)
```
````

````{tab} Class data
```python
import numpy as np
from docarray import Document,  DocumentArray

query_data = DocumentArray()
for i in range(10):
    doc = Document(
        id=str(i),
        tensor=np.array([i]),
        tags={'finetuner_label': str(i)},
    )
    query_data.append(doc)

index_data = DocumentArray()
for i in range(10):
    doc = Document(
        id=str(i + 10),
        tensor=np.array([i]),
        tags={'finetuner_label': str(i)},
    )
    index_data.append(doc)
```
````

Now we can use the evaluator. When using the euclidean distance as a matching metric, we expect to see
perfect scores, since for each query doc the nearest index doc is the one we gave as ground truth:

```python
from finetuner.tuner.evaluation import Evaluator

embed_model = EmbeddingModel()

evaluator = Evaluator(query_data, index_data, embed_model)

metrics = evaluator.evaluate(limit=1, distance='euclidean')
print(metrics)
```
```json
{
  "r_precision": 1.0,
  "precision_at_k": 1.0,
  "recall_at_k": 1.0,
  "f1_score_at_k": 1.0,
  "average_precision": 1.0,
  "hit_at_k": 1.0,
  "reciprocal_rank": 1.0,
  "dcg_at_k": 1.0,
  "ndcg_at_k": 1.0
}
```

When evaluating with a bigger matching limit, we expect precision to drop:
```python
metrics = evaluator.evaluate(limit=2, distance='euclidean')
print(metrics)
```
```json
{
  "r_precision": 1.0,
  "precision_at_k": 0.5,
  "recall_at_k": 1.0,
  "f1_score_at_k": 0.6666666666666667,
  "average_precision": 1.0,
  "hit_at_k": 1.0,
  "reciprocal_rank": 1.0,
  "dcg_at_k": 1.0,
  "ndcg_at_k": 1.0
}
```

To customize the computed metrics, an optional `metrics` argument can be provided in
the Evaluator constructor, that maps metric names to metric functions and keyword
arguments. For example:

```python
from docarray.math.evaluation import precision_at_k, recall_at_k

def f_score_at_k(binary_relevance, max_rel, k=None, beta=1.0):
    precision = precision_at_k(binary_relevance, k=k)
    recall = recall_at_k(binary_relevance, max_rel, k=k)
    return ((1+ beta**2) * precision * recall) / (beta**2 * precision + recall)

metrics = {
    'precision@5': (precision_at_k, {'k': 5}),
    'recall@5': (recall_at_k, {'k': 5}),
    'f1score@5': (f_score_at_k, {'k': 5, 'beta': 1.0})
}

evaluator = Evaluator(query_data, index_data, embed_model, metrics=metrics)
print(evaluator.evaluate(limit=2, distance='euclidean'))
```
```json
{
  "precision@5": 0.2,
  "recall@5": 1.0,
  "f1score@5": 0.33333333333333337
}
```

## Using the evaluation callback

The evaluator can be handy for computing metrics in an evaluation script, or following a `finetuner.fit`
session. For including the metrics computation in the training loop to run after each epoch, an
evaluation callback is provided. The callback can be used as follows:

```python
import finetuner
from finetuner.tuner.callback import EvaluationCallback
from docarray import DocumentArray

query_data = DocumentArray(...)
index_data = DocumentArray(...)

finetuner.fit(
    ...,
    callbacks=[EvaluationCallback(query_data, index_data)],
)

```