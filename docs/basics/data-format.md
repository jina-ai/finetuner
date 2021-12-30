(data-format)=
# Data Format

Finetuner uses Jina [`Document`](https://docs.jina.ai/fundamentals/document/) as the primitive data type. In
particular, [`DocumentArray`](https://docs.jina.ai/fundamentals/document/documentarray-api/)
and [`DocumentArrayMemap`](https://docs.jina.ai/fundamentals/document/documentarraymemmap-api/) are the input data type
in the high-level `finetuner.fit()` API. This means, your training dataset and your validation data
should be stored in `DocumentArray` or `DocumentArrayMemap`, where each training or validation instance is a `Document`
object:

```python
import finetuner

finetuner.fit(model,
              train_data=...,
              eval_data=...)
```

This chapter introduces how to construct a `Document` in a way that Finetuner will accept.

There are three different types of datasets:

```{toctree}
:maxdepth: 1

datasets/class-dataset
datasets/session-dataset
datasets/unlabeled-dataset
```

## Lazy loading

```{tip}
We recommend to keep the input `DocumentArray` as simple as possible, e.g. only fill essential fields such as `.uri`. The data loading and heavylifting work should be done via `preprocess_fn` in an on-demand manner.
```

Regardless of which dataset format we choose, all Documents need to have a non-empty `.content` attribute. Either you fill in `DocumentArray.contents` in advance, or load them on-demand. 


When working with a big dataset, loading data on-demand is more favorable as  you can not store your entire dataset into memory, . Moreover, you may want to apply random augmentations to images
each time they are used in a batch, whereas you can't prepare those augmentations in advance. For these purposes, you can use `preprocess_fn` argument of the `Dataset`. This argument is universally available from low-level `ClassDataset` and `SessionDataset` up to the top most level `finetuner.fit` API.

Let's see some examples.

### Examples

#### Load image on the fly

Let's start with an example of the first case - loading images on the fly. In this case
we would have images stored on disk, and their paths stored in the `.uri` attribute of
each `Document`. In `preprocess_fn` we would then load the image and return the numpy
array.

```python
import numpy as np
from finetuner import fit
from jina import Document, DocumentArray

dataset = DocumentArray([
  Document(uri='path/to/image.jpg', tags={'finetuner_label': 1}}),
  # ...
])

def load_image(d: Document) -> np.ndarray:
    d.load_uri_to_image_blob()
    return d.blob

model = ...
fit(model, train_data=dataset, preprocess_fn=load_image)
```

#### Image augmentation

Next, let's take a look at an example where we apply some basic image augmentation. We'll be using the [albumentations](https://albumentations.ai/) library for image augmentation in this example

```python
import albumentations as A
import numpy as np
from finetuner import fit
from jina import Document, DocumentArray

dataset = DocumentArray([
  Document(blob=np.random.rand(3, 128, 128), tags={'finetuner_label': 1}}),
  # ...
])

def augment_image(doc: Document) -> np.ndarray:
  transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
  ])
  
  return transform(image=doc.blob)['image']

model = ...
fit(model, train_data=dataset, preprocess_fn=augment_image)
```


