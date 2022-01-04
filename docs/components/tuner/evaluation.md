# Evaluation

Information retrieval metrics like NDCG and Reciprocal Rank can be calculated using the included 
{class}`~finetuner.tuner.evaluation.Evaluator` component. The evaluator can be used
[standalone](#using-the-evaluator)
or integrated in the training loop via the [evaluation callback](#using-the-evaluation-callback).


## Using the evaluator

The evaluator can be used standalone in order to compute the evaluation metrics on a sequence
of documents:
```python
from finetuner.tuner.evaluation import Evaluator
from jina import DocumentArray

query_data = DocumentArray(...)
index_data = DocumentArray(...)
embed_model = ...

evaluator = Evaluator(query_data, index_data, embed_model)

metrics = evaluator.evaluate()
```

The `query_data` (or eval data) are the documents that will be evaluated. They can be both in class 
or session format. They should contain ground truths, in the form of matches (`doc.matches`) when
using session format and in the form of labels when using class format.

If an embedding model is given, the query docs are embedded, otherwise they are assumed to carry
representations.

The `index_data` (or catalog) is an optional argument that defines the dataset against which the
query docs are matched. If not provided, query docs are matched against themselves.

The `evaluate()` method returns the computed metrics as a dictionary mapping metric names to values.
Common IR metrics, like precision, recall, average precision, dcg, ndcg and reciprocal rank are
included.


## Using the evaluation callback

The evaluator can be handy for computing metrics in an evaluation script, or following a `finetuner.fit`
session. For including the metrics computation in the training loop to run after each epoch, an
evaluation callback {class}`~finetuner.tuner.callbacks.Evaluation` is provided. The callback can be
used as follows:

```python
import finetuner
from finetuner.tuner.callback import Evaluation
from jina import DocumentArray

query_data = DocumentArray(...)
index_data = DocumentArray(...)

finetuner.fit(
    ...,
    callbacks=[Evaluation(query_data, index_data)],
)

```