(session-dataset)=
# Session Dataset

A {class}`~fintuner.tuner.dataset.SessionDataset` is a `DocumentArray`, in which each Document has `matches`. The labels are stored under matched Document's `.tags['finetuner_label']`. The label can be either `1` (denoting similarity of match to its parent `Document`) or `-1` (denoting dissimilarity of the match from its parent `Document`).

The word "session" comes from the scenario where a user's click-through introduces an implicit yes/no on the relevance, hence an interactive session. 

## Batch construction

A `SessionDataset` works with {class}`~fintuner.tuner.dataset.SessionSampler`. 

Here the batches are simply constructed by putting together enough root documents and their matches (we call this a *session*) to fill the batch according to the `batch_size` parameter. An example of a batch of size `batch_size=8` made of two sessions is shown on the image below. `0` denotes the root document.

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
    [Document(content='shirt'), Document(content='shoe')]
)

ds[0].matches = [
    Document(content='red shirt', tags={'finetuner_label': 1}),
    Document(content='red shoe', tags={'finetuner_label': -1}),
]
ds[1].matches = [
    Document(content='black shoe', tags={'finetuner_label': 1}),
    Document(content='black pants', tags={'finetuner_label': -1}),
]

sds = SessionDataset(ds)
for b in SessionSampler(sds.labels, batch_size=2):
    print([sds[bb] for bb in b])
```


```text
[('shirt', (0, 0)), ('red shoe', (0, -1))]
[('shoe', (1, 0)), ('black pants', (1, -1))]
```

We got 2 batches here.

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

One can use {meth}`~finetuner.toydata.generate_qa` to generate it.