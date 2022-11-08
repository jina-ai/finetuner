(create-training-data)=
# Prepare Training Data

Finetuner accepts training data and evaluation data in the form of CSV files 
or {class}`~docarray.array.document.DocumentArray` objects.
Because Finetuner follows a [supervised-learning](https://en.wikipedia.org/wiki/Supervised_learning) scheme, each element requires a label that identifies which other elements it should be similar to. 
If you need to evaluate metrics on separate evaluation data, it is recommended to create a dataset only for evaluation purposes. This can be done in the same way a a training dataset is created, as described below.

Data can be prepared in two different formats, either as a CSV file, or as a {class}`~docarray.array.document.DocumentArray`. In the sections below, you can see examples which demonstrate how the training datasets should look like for each format.

## Preparing CSV Files

To record data in a CSV file, the contents of each element are stored plainly, with each row either representing one labeled item, a pair of items that should be semantically similar, or two items of different modalities in the case that a CLIP model is being used. The provided CSV files are then parsed and a {class}`~docarray.array.document.DocumentArray` is constructed containing the elements within the CSV file.
Currently, `excel`, `excel-tab` and `unix` CSV dialects are supported. To specify which dialect to use, provide a {class}`~finetuner.data.CSVOptions` object with `dialect=chosen_dialect` as the `csv_options` argument to the {meth}`~finetuner.fit` function. The list of all options for reading CSV files can be found in the description of the {class}`~finetuner.data.CSVOptions` class.

````{tab} Labeled data
In cases where you want multiple elements grouped together, you can provide a label in the second column. This way, all elements in the first column that have the same label will be considered similar when training. To indicate that the second column of your CSV file represents a label instead of a second element, set `is_labeled = True` in the `csv_options` argument of the {meth}`~finetuner.fit` function. Your data can then be structured like so:

```markdown
Hello!                  greeting-english
Hi there.               greeting-english
Good morning.           greeting-english
I'm (…) sorry!          apologize-english
I'm sorry to have…      apologize-english
Please, forgive me!     apologize-english
```

When using image-to-image retrieval models, images can be represented as a URI or a path to a file:

```markdown
/Users/images/apples/green_apple.jpg    picture of apple
/Users/images/apples/red_apple.jpg      picture of apple
https://example.com/apple-styling.jpg   picture of apple
/Users/images/oranges/orange.jpg        picture of orange
https://example.com/orange-styling.jpg  picture of orange
```

```diff
import finetuner
from finetuner import CSVOptions

run = finetuner.fit(
    ...,
    train_data='your-data.csv',
-   csv_options=CSVOptions(),
+   csv_options=CSVOptions(is_labeled=True)
)
```

```{important} 
If paths to local images are provided,
they can be loaded into memory by setting `convert_to_blob = True` in the {class}`~finetuner.data.CSVOptions` object.
It is worth noting that this setting does not cause Internet URLs to be loaded into memory.
```

````

````{tab} text-to-image search using CLIP
To prepare data for text-to-image search, each row must contain one uri to an image, and one piece of text. The order that these two are placed does not matter, so long as the ordering is kept consistent for all rows.

```markdown
This is a photo of an apple.                apple.jpg
This is a black-white photo of an organge.  orange.jpg
```

```{admonition} CLIP model explained
:class: hint
OpenAI CLIP model wraps two models: a vision transformer and a text transformer.
During fine-tuning, we're optimizing two models in parallel.

At the model saving time, you will discover, we are saving two models to your local directory. 
```

````


````{tab} two elements per row
If you want two elements to be semantically close together, they can be placed on the same row as a pair, Each pair will automatically be assigned a distinct label:

```markdown
This is an English sentence         Das ist ein englischer Satz
This is another English sentence    Dies ist ein weiterer englischer Satz
...
```

```{warning}
When reading in files constructed in this way, a unique label is generated for each pair. Some sampling methods that require more than two elements per class will not work because each class in the dataset will only contain a single pair of elements.


```

```{warning}
Please remove/replace comma in your data fields if you are using a comma `,` as a delimiter.


```

````

It is worth noting that we support the following dialects:

+ `excel` use `,` as delimiter and `\r\n` as lineterminator.
+ `excel-tab` use `\t` as delimiter and `\r\n` as lineterminator.
+ `unix` use `,` as delimiter and `\n` as lineterminator.


## Preparing a DocumentArray
When providing training data in a DocumentArray, each element is represented as a {class}`~docarray.document.Document`. You should assign a label to each {class}`~docarray.document.Document` inside your {class}`~docarray.array.document.DocumentArray`.
For most of the models, this is done by adding a `finetuner_label` tag to each document. {class}`~docarray.document.Document`s containing uris that point to local images can load these images into memory using the {meth}`docarray.document.Document.load_uri_to_blob` function of that {class}`~docarray.document.Document`.
Only for cross-modality (text-to-image) fine-tuning with CLIP, is this not necessary as explained at the bottom of this section.


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
During the training, CLIP learns to place documents that are part of a pair close to
each other and documents that are not part of a pair far from each other.
As a result, no further labels need to be provided.
