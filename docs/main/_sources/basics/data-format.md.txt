(data-format)=
# Data Format

Finetuner uses Jina [`Document`](https://docs.jina.ai/fundamentals/document/) as the primitive data type. In
particular, [`DocumentArray`](https://docs.jina.ai/fundamentals/document/documentarray-api/)
and [`DocumentArrayMemap`](https://docs.jina.ai/fundamentals/document/documentarraymemmap-api/) are the input data type
in the high-level `finetuner.fit()` API. This means, your training dataset and evaluation dataset should be stored in `DocumentArray`
or `DocumentArrayMemap`, where each training or evaluation instance is a `Document` object:

```python
import finetuner

finetuner.fit(model,
              train_data=...,
              eval_data=...)
```

This chapter introduces how to construct a `Document` in a way that Finetuner will accept.

There are three different types of datasets:
- [Unlabeled dataset](#unlabeled-dataset)
- [Class dataset](#class-dataset) (a labeled dataset)
- [Session dataset](#session-dataset) (a labeled dataset)

## Understand supervision

Finetuner tunes a deep neural network on search tasks. In this context, the supervision comes from whether the nearest-neighbour matches are good or bad, where matches are often computed on model's embeddings. You have to label these good matches and bad matches so Finetuner can learn your feedback and improve the model. The following graph illustrates the process:

```{figure} tuner-journey.svg
:align: center
```

## Unlabeled dataset

When using `finetuner.fit(..., interactive=True)`, you only need to provide a `DocumentArray`-like object where each `Document` object contains `.content`. This is because Finetuner will start a web frontend for interactive labeling. Hence, the supervision comes directly from you.

All that the `Document`s in this dataset need is a `.content` attribute (which can be `text`, `blob` or `uri` - if you implement [your own loading](#loading-and-preprocessing)).

(construct-labeled-data)=
## Labeled datasets

When using `finetuner.fit(..., interactive=False)`, you must provide some kind of labels
to your documents, so that the model can learn. There are two different kinds of labeled datasets that finetuner supports: a class dataset and a session dataset.

Regardless of which dataset format we choose, all documents need a `.content` attribute (which can be `text`, `blob` or `uri` - if you implement [your own loading](#loading-and-preprocessing)).

### Class dataset

In this dataset, each `Document` in the dataset has a "class" label stored under `.tags['finetuner']['label']`. The document structure is flat - no `matches` are needed.

The "class" label is not necessarily related to classification - rather, it is there to denote similar `Document`s. All `Document`s with the same label are considered similar, and all `Document`s with different labels are considered dis-similar.

This "class" label is used to construct batches for model training. In each batch, a number of different classes will be randomly selected, and from each class a number of `Document`s will be taken, so as to fill the batch.

Specifically, the size of the batch is controlled by the `batch_size` argument, and number of instances to take from each class is controlled by the `num_items_per_class`. The number of instances per batch is computed dynamically, so that the batch is full. The image below illustrates this for `batch_size=8` and `num_items_per_class=2`.

```{figure} class-dataset.png
:align: center
```

Here is an example of a class dataset

```python
from jina import Document, DocumentArray

class_data = DocumentArray([
  Document(text='some text for class 1', tags={'finetuner': {'label': 1}}),
  Document(text='more text for class 1', tags={'finetuner': {'label': 1}}),
  Document(text='some text for class 2', tags={'finetuner': {'label': 2}}),
  Document(text='more text for class 2', tags={'finetuner': {'label': 2}}),
])
```

### Session dataset

In this dataset, each root `Document` in the dataset has `matches`, but no label. Instead, its matches have a label stored under `.tags['finetuner']['label']`. This label can be either `1` (denoting similarity of match to its reference `Document`) or `-1` (denoting dissimilarity of the match from its reference `Document`).

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

This dataset is an example of class dataset - each `Document` has a class label (corresponding to one of the 10 classes). The `Document`s contain the following relevant information:

- `.blob`: the numpy array of the image
- `.tags['finetuner']['label']`: the class label


(build-qa-data)=
### Covid QA

Covid QA data is a CSV that has 481 rows with the columns `question`, `answer` & `wrong_answer`. 

```{figure} covid-qa-data.png
:align: center
```

To convert this dataset into match data, we build each Document to contain the following relevant information:

- `.text`: the `question` column
- `.matches`: the generated positive & negative matches Document
    - `.text`: the `answer`/`wrong_answer` column
    - `.tags['finetuner']['label']`: the match label: `1` or `-1`.

Matches are built with the logic below:

- only allows 1 positive match per Document, it is taken from the `answer` column;
- always include `wrong_answer` column as the negative match. Then sample other documents' answer as negative matches.


```{tip}

The Finetuner codebase contains two synthetic matching data generators for demo and debugging purpose:

- `finetuner.toydata.generate_fashion()`: the generator of Fashion-MNIST matching data.
- `finetuner.toydata.generate_qa()`: the generator of Covid QA matching data.

```
