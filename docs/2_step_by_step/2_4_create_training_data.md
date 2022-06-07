# Create training and evaluation (Optional) data

Finetuner accepts training data and evaluation data in the form of [docarray](https://docarray.jina.ai/) `DocumentArray` objects.
Because Finetuner follows a [supervised-learning](https://en.wikipedia.org/wiki/Supervised_learning) scheme,
you should assign a label to each `Document` inside your `DocumentArray` as follows:

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
        content='https://...skirt-1.png',
        tags={'finetuner_label': 'skirt'},
    ),
    Document(
        content='https://...t-shirt-1.png',
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
                tags={'finetuner_label': 'skirt-1'}
            ),
            Document(
                content='https://...skirt-1.png',
                modality='image',
                tags={'finetuner_label': 'skirt-1'}
            ),
        ],
    ),
    Document(
        chunks=[
            Document(
                content='stripped over-sized shirt for sell',
                modality='text',
                tags={'finetuner_label': 'shirt-1'}
            ),
            Document(
                content='https://...shirt-1.png',
                modality='image',
                tags={'finetuner_label': 'shirt-1'}
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
you should create a root `Document` which wraps two `chunks` with the `image` and `text` modality.
The image and text form a pair.

```{admonition} CLIP model explained
:class: hint
OpenAI CLIP model wraps two models: a vision transformer and a text transformer.
During fine-tuning, we're optimizing two models in parallel.

At the model saving time, you will discover, we are saving two models to your local directory. 
```

 If you need to evaluate metrics on separate evaluation data,
It is recommended to create another `DocumentArray` as above only for evaluation purposes.

Carry on, you're almost there!
