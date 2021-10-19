(data-format)=
# Data Format

Finetuner uses Jina [`Document`](https://docs.jina.ai/fundamentals/document/) as the primitive data type. In
particular, [`DocumentArray`](https://docs.jina.ai/fundamentals/document/documentarray-api/)
and [`DocumentArrayMemap`](https://docs.jina.ai/fundamentals/document/documentarraymemmap-api/) are the input data type
for Tailor and Tuner. This means, your training dataset and evaluation dataset should be stored in `DocumentArray`
or `DocumentArrayMemap`, where each training or evaluation instance is a `Document` object.

This chapter introduces how to construct a `Document` in a way that Finetuner will accept.

## Understand supervision

Finetuner tunes a deep neural network on search tasks. In this context, the supervision comes from whether the nearest-neighbour matches are good or bad, where matches are often computed on model's embeddings. You have to label these good matches and bad matches so Finetuner can learn your feedback and improve the model. The following graph illustrates the process:

```{figure} tuner-journey.svg
:align: center
```

## Required fields

When using `finetuner.fit(..., interactive=True)`, you only need to provide a `DocumentArray`-like object where each `Document` object contains `.content`. This is because Finetuner will start a web frontend for interactive labeling. Hence, the supervision comes directly from you.

(construct-labeled-data)=
When using `finetuner.fit(..., interactive=False)`, your `Document` object needs to contain:

- [`.content`](https://docs.jina.ai/fundamentals/document/document-api/#document-content): can be `.blob`, `.text`
  or `.buffer`;
- at least one `Document`
  in [`.matches`](https://docs.jina.ai/fundamentals/document/document-api/#recursive-nested-document), and
  each `Document` in `.matches` needs to contain
    - `.content`: it should be the same data type as its parent `Document`;
    - `.tags['finetuner']['label']`.

In summary, you either label the matches on-the-fly or prepare the labeled data in advance.

### Matches

Finetuner relies on matching data in `.matches`. To manually add a match to a `Document` object, you can do:

```python
from jina import Document

d = Document(text='hello, world!')
m = Document(text='hallo, welt!')

d.matches.append(m)

print(d)
```

```text
{'id': '67432a92-1f9f-11ec-ac8a-1e008a366d49', 'matches': [{'id': '67432cd6-1f9f-11ec-ac8a-1e008a366d49', 'mime_type': 'text/plain', 'text': 'hallo, welt!', 'adjacency': 1}], 'mime_type': 'text/plain', 'text': 'hello, world!'}
```

Note that the match `Document` should share the same content type as its parent `Document`. The following combinations
are not valid in Finetuner:

```python
from jina import Document
import numpy as np

d = Document(text='hello, world!')
m1 = Document(buffer=b'h236cf4')
m2 = Document(blob=np.array([1, 2, 3]))

d.matches.append([m1, m2])
```

If you have two `DocumentArray` each of which is filled with `.embeddings`, then you can simply call `.match()` function
to build matches for every Document in one-shot:

```python
from jina import DocumentArray, Document
import numpy as np

da1 = DocumentArray([Document(text='hello, world!', embedding=np.array([1, 2, 3])),
                     Document(text='goodbye, world!', embedding=np.array([4, 5, 6]))])

da2 = DocumentArray([Document(text='hallo, welt!', embedding=np.array([1.5, 2.5, 3.5])),
                     Document(text='auf wiedersehen, welt!', embedding=np.array([4.5, 5.5, 6.5]))])

da1.match(da2)

print(da1)
```

```text
DocumentArray has 2 items:
{'id': 'a5dd3158-1f9f-11ec-9a49-1e008a366d49', 'matches': [{'id': 'a5dd3b94-1f9f-11ec-9a49-1e008a366d49', 'mime_type': 'text/plain', 'text': 'hallo, welt!', 'embedding': {'dense': {'buffer': 'AAAAAAAA+D8AAAAAAAAEQAAAAAAAAAxA', 'shape': [3], 'dtype': '<f8'}}, 'adjacency': 1, 'scores': {'cosine': {'value': 0.002585097}}}, {'id': 'a5dd3d74-1f9f-11ec-9a49-1e008a366d49', 'mime_type': 'text/plain', 'text': 'auf wiedersehen, welt!', 'embedding': {'dense': {'buffer': 'AAAAAAAAEkAAAAAAAAAWQAAAAAAAABpA', 'shape': [3], 'dtype': '<f8'}}, 'adjacency': 1, 'scores': {'cosine': {'value': 0.028714137}}}], 'mime_type': 'text/plain', 'text': 'hello, world!', 'embedding': {'dense': {'buffer': 'AQAAAAAAAAACAAAAAAAAAAMAAAAAAAAA', 'shape': [3], 'dtype': '<i8'}}},
{'id': 'a5dd3784-1f9f-11ec-9a49-1e008a366d49', 'matches': [{'id': 'a5dd3d74-1f9f-11ec-9a49-1e008a366d49', 'mime_type': 'text/plain', 'text': 'auf wiedersehen, welt!', 'embedding': {'dense': {'buffer': 'AAAAAAAAEkAAAAAAAAAWQAAAAAAAABpA', 'shape': [3], 'dtype': '<f8'}}, 'adjacency': 1, 'scores': {'cosine': {'value': 0.00010502179}}}, {'id': 'a5dd3b94-1f9f-11ec-9a49-1e008a366d49', 'mime_type': 'text/plain', 'text': 'hallo, welt!', 'embedding': {'dense': {'buffer': 'AAAAAAAA+D8AAAAAAAAEQAAAAAAAAAxA', 'shape': [3], 'dtype': '<f8'}}, 'adjacency': 1, 'scores': {'cosine': {'value': 0.011804931}}}], 'mime_type': 'text/plain', 'text': 'goodbye, world!', 'embedding': {'dense': {'buffer': 'BAAAAAAAAAAFAAAAAAAAAAYAAAAAAAAA', 'shape': [3], 'dtype': '<i8'}}}
```

```{tip}
The field `.embedding` is not required by Finetuner.
```

### Labels

The label represents how related a `match` is to the `Document`. It is stored in `.tags['finetuner']['label']`.

In Finetuner, float number `1` is considered as a positive relation between `match` and `Document`; whereas float
number `-1` is considered as no relation between `match` and `Document`.

For example, we have four sentences:

```text
hello, world!
hallo, welt!
Bonjour, monde!
goodbye, world!
```

Now to construct the matches of `Document(text='hello, world!')` for expressing that texts in different languages are
more related to this Document than `goodbye, world`:

```python
from jina import Document

d = Document(text='hello, world!')
m1 = Document(text='hallo, welt!', tags={'finetuner': {'label': 1}})
m2 = Document(text='Bonjour, monde!', tags={'finetuner': {'label': 1}})
m3 = Document(text='goodbye, world!', tags={'finetuner': {'label': -1}})

d.matches.extend([m1, m2, m3])
```

```text
{'id': '0e7ec5aa-1faa-11ec-a46a-1e008a366d49', 'matches': [{'id': '0e7ec7c6-1faa-11ec-a46a-1e008a366d49', 'mime_type': 'text/plain', 'tags': {'finetuner': {'label': 1.0}}, 'text': 'hallo, welt!', 'adjacency': 1}, {'id': '0e7ecd52-1faa-11ec-a46a-1e008a366d49', 'mime_type': 'text/plain', 'tags': {'finetuner': {'label': 1.0}}, 'text': 'Bonjour, monde!', 'adjacency': 1}, {'id': '0e7ece7e-1faa-11ec-a46a-1e008a366d49', 'mime_type': 'text/plain', 'tags': {'finetuner': {'label': -1.0}}, 'text': 'goodbye, world!', 'adjacency': 1}], 'mime_type': 'text/plain', 'text': 'hello, world!'}
```

```{admonition} Is it okay to have all matches as 1, or all as -1?
:class: hint

Yes. Labels should reflect the groundtruth as-is. If a Document contains only positive matches or only negative matches, then so be it.

However, if all match labels from all Documents are the same, then Finetuner cannot learn anything useful.
```

### Catalog

In search, queries and search results are often distinct sets.
Specifying a `catalog` helps you keep this distinction during finetuning.
When using `finetuner.fit(train_data=...,eval_data=..., catalog=...)`, `train_data` and `eval_data` specify the potential queries and the `catalog` specifies the potential results.
This distinction is mainly used

- in the Labeler, when new sets of unlabeled results are generated and
- during evaluation, for the NDCG calculation.

A `catalog` is either a `DocumentArray` or a `DocumentArrayMemmap`.
If no `catalog` is specified, the Finetuner will implicitly use `train_data` as catalog.

## Data source

After organizing the labeled `Document` into `DocumentArray` or `DocumentArrayMemmap`, you can feed them
into `finetuner.fit()`.

But where do the labels come from? You can use Labeler, which allows you to interactively label data and tune the model at
the same time.

Otherwise, you will need to prepare labeled data on your own.

## Examples

Here are some examples for generating synthetic matching data for Finetuner. You can learn how the constructions are
made here.

(build-mnist-data)=
### Fashion-MNIST

Fashion-MNIST contains 60,000 training images and 10,000 images in 10 classes. Each image is a single channel 28x28
grayscale image.


```{figure} fashion-mnist-sprite.png
:align: center
```

To convert this dataset into match data, we build each Document to contain the following relevant information:

- `.blob`: the image;
- `.matches`: the generated positive and negative matches of the Document;
    - `.blob`: the matched Document's image;
    - `.tags['finetuner']['label']`: the match label: `1` or `-1`.

Matches are built with the logic below:

- randomly sample same-class Documents as positive matches, i.e. labeled with `1`;
- randomly sample other-class Documents as negative matches, i.e. labeled with `-1`.

(build-qa-data)=
### Covid QA



Covid QA data is a CSV that has 481 rows with the columns `question`, `answer` & `wrong_answer`. 

```{figure} covid-qa-data.png
:align: center
```

To convert this dataset into match data, we build each Document to contain the following relevant information:

- `.text`: the original `question` column
- `.blob`: a fixed length `ndarray` tokenized from `.text`
- `.matches`: the generated positive & negative matches Document
    - `.text`: the original `answer`/`wrong_answer` column
    - `.blob`: a fixed length `ndarray` tokenized from `.text`
    - `.tags['finetuner']['label']`: the match label: `1` or `-1`.

Matches are built with the logic below:

- only allows 1 positive match per Document, it is taken from the `answer` column;
- always include `wrong_answer` column as the negative match. Then sample other documents' answer as negative matches.


```{tip}

The Finetuner codebase contains two synthetic matching data generators for demo and debugging purpose:

- `finetuner.toydata.generate_fashion_match()`: the generator of Fashion-MNIST matching data.
- `finetuner.toydata.generate_qa_match()`: the generator of Covid QA matching data.

```
