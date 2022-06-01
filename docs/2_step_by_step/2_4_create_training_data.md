# Create training and evaluation (Optional) data

Finetuner accepts [docarray](https://docarray.jina.ai/) `DocumentArray` as training data and evaluation data.
Because Finetuner follows a [supervised-learning](https://en.wikipedia.org/wiki/Supervised_learning) scheme,
you should assign label to each `Document` inside `DocumentArray` with:

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

As shown in the above code blocks.
For single-modality model fine-tuning, you only need to create a `Document` with `content` and `tags` with the `finetuner_label`.

For cross-modality (text-to-image) fine-tuning with CLIP,
you should create a root `Document` which wraps two `chunks` with the `image` and `text` modality.
The image and text form a pair.

```{admonition} CLIP model explained
:class: hint
OpenAI CLIP model wraps two models: a vision transformer and a text transformer.
During fine-tuning, we're optimizing two models in parallel.

At the model saving time, you will discover we are saving two models to your local directory. 
```

If you need to evaluate metrics on a seperate evaluation data,
you're recommended to create another `DocumentArray` as above only for evaluation purpose.

```{admonition} What happened underneath?
:class: hint
Finetuner will push your training data and evaluation data to the cloud storage.
During fine-tuning, your DocumentArray will be converted into Pytorch DataLoader using Distributed Data Parallel. 
```

Carry on, you're almost there!
