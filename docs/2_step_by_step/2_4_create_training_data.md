# Create training and evaluation (Optional) data

Finetuner accept [docarray](https://docarray.jina.ai/) `DocumentArray` as training data and evaluation data.
Because Finetuner follows a [supervised-learning](https://en.wikipedia.org/wiki/Supervised_learning) scheme,
you should assign label to each `Document` inside `DocumentArray` with:

```python
from docarray import Document, DocumentArray

train_da = DocumentArray([
    Document(content=..., tags={'finetuner_label': 'skirt'}),
    Document(content=..., tags={'finetuner_label': 't-shirt'}),
    ...,
])
```

````{tab} text-to-text search
```python
from docarray import Document, DocumentArray

train_da = DocumentArray([
    Document(content='pencil skirt slim fit available for sell', tags={'finetuner_label': 'skirt'}),
    Document(content='stripped over-sized shirt for sell', tags={'finetuner_label': 't-shirt'}),
    ...,
])
```
````
````{tab} image-to-image search
```python
from docarray import Document, DocumentArray

train_da = DocumentArray([
    Document(content='https://...skirt-1.png', tags={'finetuner_label': 'skirt'}),
    Document(content='https://...t-shirt-1.png', tags={'finetuner_label': 't-shirt'}),
    ...,
])
```
````
````{tab} text-to-image search on CLIP
```python
from docarray import Document, DocumentArray

train_da = DocumentArray([
    Document(
        chunks=[
            Document(content='pencil skirt slim fit available for sell', modality='text', tags={'finetuner_label': 'skirt-1'}),
            Document(content='https://...skirt-1.png', modality='image', tags={'finetuner_label': 'skirt-1'}),
        ],
    ),
    Document(
        chunks=[
            Document(content='stripped over-sized shirt for sell', modality='text', tags={'finetuner_label': 'shirt-1'}),
            Document(content='https://...shirt-1.png', modality='image', tags={'finetuner_label': 'shirt-1'}),
        ],
    ),
])
```
````
