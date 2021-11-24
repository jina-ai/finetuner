(session-dataset)=
# Session Dataset

A {class}`~fintuner.tuner.dataset.SessionDataset` is a `DocumentArray`, in which each Document has `matches` Document. The labels are stored under matched Document's `.tags['finetuner_label']`. The label can be either `1` (denoting similarity of match to its parent `Document`) or `-1` (denoting dissimilarity of the match from its parent `Document`).

The word "session" comes from the scenario where a user's click-through introduces an implicit yes/no on the relevance, hence an interactive session. 

## Batch construction

A `SessionDataset` works with {class}`~fintuner.tuner.dataset.SessionSampler`. 

Here the batches are simply constructed by putting together enough root documents and their matches (we call this a *session*) to fill the batch according to `batch_size` parameter. An example of a batch of size `batch_size=8` made of two sessions is show on the image below.

```{figure} ../session-dataset.png
:align: center
```


## Examples

### Toy example


Here is an example of a session dataset

```python
from jina import Document, DocumentArray

from finetuner.tuner.dataset import SessionDataset, SessionSampler

ds = DocumentArray(
    [Document(id='0'), Document(id='1')]
)

ds[0].matches = [
    Document(id='2', tags={'finetuner_label': 1}),
    Document(id='3', tags={'finetuner_label': -1}),
]
ds[1].matches = [
    Document(id='4', tags={'finetuner_label': 1}),
    Document(id='5', tags={'finetuner_label': -1}),
]

for b in SessionSampler(SessionDataset(ds).labels, batch_size=2):
    print(b)
```


```text
[0, 2]
[3, 4]
```

(build-qa-data)=
### Covid QA data

Covid QA data is a CSV that has 481 rows with the columns `question`, `answer` & `wrong_answer`. 

```{figure} ../covid-qa-data.png
:align: center
```

To convert this dataset into a `SessionDataset`, we build each Document to contain the following relevant information:

- `.text`: the `question` column
- `.matches`: the generated positive & negative matches Document
    - `.text`: the `answer`/`wrong_answer` column
    - `.tags['finetuner_label']`: the match label: `1` or `-1`.

Matches are built with the logic below:

- only allows 1 positive match per Document, it is taken from the `answer` column;
- always include `wrong_answer` column as the negative match. Then sample other documents' answer as negative matches.