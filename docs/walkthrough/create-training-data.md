(create-training-data)=
# Prepare Training Data

Finetuner accepts training data and evaluation data in the form of {class}`~docarray.array.document.DocumentArray` objects.
Because Finetuner follows a [supervised-learning](https://en.wikipedia.org/wiki/Supervised_learning) scheme,
you should assign a label to each {class}`~docarray.document.Document` inside your {class}`~docarray.array.document.DocumentArray` as follows:

````{tab} text-to-text search
```python
from docarray import Document, DocumentArray

train_da = DocumentArray([
    Document(
        content='pencil skirt slim fit available for sell',
        tags={'finetuner_label': 'skirt'}
    ),
    Document(
        content='stripped over-sized shirt for sell',
        tags={'finetuner_label': 't-shirt'}
    ),
    ...,
])
```
````
````{tab} image-to-image search
```python
from docarray import Document, DocumentArray

train_da = DocumentArray([
    Document(
        uri='https://...skirt-1.png',
        tags={'finetuner_label': 'skirt'},
    ),
    Document(
        uri='https://...t-shirt-1.png',
        tags={'finetuner_label': 't-shirt'},
    ),
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
            Document(
                content='pencil skirt slim fit available for sell',
                modality='text',
            ),
            Document(
                uri='https://...skirt-1.png',
                modality='image',
            ),
        ],
    ),
    Document(
        chunks=[
            Document(
                content='stripped over-sized shirt for sell',
                modality='text',
            ),
            Document(
                uri='https://...shirt-1.png',
                modality='image',
            ),
        ],
    ),
])
```
````

As was shown in the above code blocks,
when fine-tuning a model with a single modality (e.g. image),
you only need to create a `Document` with `content` and `tags` with the `finetuner_label`.

For cross-modality (text-to-image) fine-tuning with CLIP,
you should create a root `Document` which wraps two {class}`~docarray.array.chunk`s with the `image` and `text` modality.
The image and text form a pair.
Evaluation data should be created the same way as above.

```{admonition} CLIP model explained
:class: hint
OpenAI CLIP model wraps two models: a vision transformer and a text transformer.
During fine-tuning, we're optimizing two models in parallel.

At the model saving time, you will discover, we are saving two models to your local directory. 
```

 If you need to evaluate metrics on separate evaluation data,
It is recommended to create another `DocumentArray` as above only for evaluation purposes.

Carry on, you're almost there!
