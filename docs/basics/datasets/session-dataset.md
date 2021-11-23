# Session dataset

In this dataset, each root `Document` in the dataset has `matches`, but no label. Instead, its matches have a label stored under `.tags['finetuner_label']`. This label can be either `1` (denoting similarity of match to its reference `Document`) or `-1` (denoting dissimilarity of the match from its reference `Document`).

This dataset is meant for cases where the relationship is only known between a small subset of documents. For example, our data could come from a search engine: given a query (which would be a root document), the results that the users clicked on are considered similar to the query (and will thus be a match with label `1`), while the results that the users did not click on are deemed dissimilar (and will thus be labeled with `-1`). Note that no assumption is made about the similarity between the "dissimilar" (labeled with `-1`) items themselves.

Here the batches are simply constructed by putting together enough root documents and their matches (we call this a *session*) to fill the batch according to `batch_size` parameter. An example of a batch of size `batch_size=8` made of two sessions is show on the image below.

```{figure} session-dataset.png
:align: center
```

Here is an example of a session dataset

```python
from jina import Document, DocumentArray

root_documents = DocumentArray(
  [Document(text='trousers'), Document(text='polo shirt')]
)

root_documents[0].matches = [
  Document(text='shorts', tags={'finetuner': {'label': 1}}),
  Document(text='blouse', tags={'finetuner': {'label': -1}}),
]
root_documents[1].matches = [
  Document(text='t-shirt', tags={'finetuner': {'label': 1}}),
  Document(text='earrings', tags={'finetuner': {'label': -1}}),
]
```

## Loading and preprocessing

There are cases when you can not store your entire dataset into memory, and need to load
individual items on the fly. Similarly, you may want to apply random augmentations to images
each time they are used in a batch. For these purposes, you can pass a pre-processing function
using `preprocess_fn` argument to the `fit` function.

Let's start with an example of the first case - loading images on the fly. In this case
we would have images stored on disk, and their paths stored in the `.uri` attribute of
each `Document`. In `preprocess_fn` we would then load the image and return the numpy
array.

```python
import numpy as np
from finetuner import fit
from jina import Document, DocumentArray
from PIL import Image

dataset = DocumentArray([
  Document(uri='path/to/image.jpg', tags={'finetuner': {'label': 1}}),
  # ...
])

def load_image(path: str) -> np.ndarray:
  image = Image.open(path)
  return np.array(image)

model = ...
fit(model, train_data=dataset, preprocess_fn=load_image)
```

Next, let's take a look at an example where we apply some basic image augmentation. We'll be using the [albumentations](https://albumentations.ai/) library for image augmentation in this example

```python
import albumentations as A
import numpy as np
from finetuner import fit
from jina import Document, DocumentArray

dataset = DocumentArray([
  Document(blob=np.random.rand(3, 128, 128), tags={'finetuner': {'label': 1}}),
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
