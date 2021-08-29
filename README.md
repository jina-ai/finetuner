# Finetuner

## Dev Install

Finetuner requires your local `jina` as the latest master. It is the best if you have `jina` installed
via `pip install -e .`.

```bash
git clone https://github.com/jina-ai/finetuner.git
cd finetuner
# pip install -r requirements.txt (only required when you do not have jina locally) 
pip install -e .
```

#### Install tests requirements

```bash
pip install -r ./github/requirements-test.txt
pip install -r ./github/requirements-cicd.txt
```

#### Enable precommit hook

The codebase is enforced with Black style, please enable precommit hook.

```bash
pre-commit install
```

## Usage

```python
import finetuner

finetuner.fit(...)
```

## Examples

### Tune a simple MLP on Fashion-MNIST

1. Write a base model. A base model can be written in Keras/Pytorch/Paddle. It can be either a new model or an existing model with pretrained weights. Below we construct a `784x128x32` MLP from scratch.

    - in Keras:
        ```python
        import tensorflow as tf
        base_model = tf.keras.Sequential([
                    tf.keras.layers.Flatten(input_shape=(28, 28)),
                    tf.keras.layers.Dense(128, activation='relu'),
                    tf.keras.layers.Dense(32)
                ])
        ```

    - in Pytorch:
        ```python
        import torch
        base_model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(
                in_features=28 * 28,
                out_features=128,
            ),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=128, out_features=32))
        ```

    - in Paddle:
        ```python
        import paddle
        base_model = paddle.nn.Sequential(
            paddle.nn.Flatten(),
            paddle.nn.Linear(
                in_features=28 * 28,
                out_features=128,
            ),
            paddle.nn.ReLU(),
            paddle.nn.Linear(in_features=128, out_features=32))
        ```

2. Call `finetune.fit` on the base model and match data:

    ```python
    import finetuner
    from tests.data_generator import fashion_match_doc_generator as mdg

    finetuner.fit(
        base_model,
        head_layer='CosineLayer',
        train_data=mdg,
        eval_data=lambda: mdg(is_testset=True)
    )
    ```

## Generate Synthetic Match Data

We use Fashion-MNIST and Covid QA data for generating synthetic matching data, as these two datasets align with the
first two `jina hello` demos.

- `tests.data_generator.fashion_match_doc_generator()`: the generator of Fashion-MNIST synthetic matching data.
- `tests.data_generator.qa_match_doc_generator()`: the generator of Fashion-MNIST synthetic matching data.

### Fashion-MNIST as synthetic matching data

Fashion-MNIST contains 60,000 training images and 10,000 images in 10 classes. Each image is a single channel 28x28
grayscale image. To convert this dataset into match data, we build each document to contain the following info that are
relevant:

- `.blob`: the image
- `.matches`: the generated positive & negative matches Document
    - `.blob`: the matched Document's image
    - `.tags['finetuner']['label']`: the match label, can be `1` or `-1` or user-defined.

Matches are built with the logic below:
- sample same-class documents as positive matches; - sample other-class documents as negative matches.

### Covid QA as synthetic matching data

Covid QA data is a CSV that has 481 rows with columns `question`, `answer` & `wrong_answer`. To convert this dataset
into match data, we build each document to contain the following info that are relevant:

- `.text`: the original `question` column
- `.blob`: when set `to_ndarray` to True, `text` is tokenized into a fixed length `ndarray`.
- `.matches`: the generated positive & negative matches Document
    - `.text`: the original `answer`/`wrong_answer` column
    - `.tags['finetuner']['label']`: the match label, can be `1` or `-1` or user-defined.
    - `.blob`: when set `to_ndarray` to True, `text` is tokenized into a fixed length `ndarray`.

Matches are built with the logic below:
- only allows 1 positive match per Document, it is taken from the `answer` column. - always include `wrong_answer`
column as the negative match. Then sample other documents' answer as negative matches.

### Generator API

```python
from tests.data_generator import fashion_match_doc_generator as mdg

# or

from tests.data_generator import qa_match_doc_generator as mdg
```

#### To get only first 10 documents

```python

for d in mdg(num_total=10):
    ...
```

#### To set number of positive/negative samples per document

```python
for d in mdg(num_pos=2, num_neg=7):
    ...
```

`qa_match_doc_generator` has a fixed number of positive matches `1`.

#### To set the label value of positive & negative samples

```python
for d in mdg(pos_value=1, neg_value=-1):
    ...
```

#### To make image as 3-channel pseudo RGB image

```python
from tests.data_generator import fashion_match_doc_generator as fmdg

for d in fmdg(channels=3):
    ...
```

#### To upsample image as 112x112 3-channel pseudo RGB image

```python
from tests.data_generator import fashion_match_doc_generator as fmdg

for d in fmdg(channels=3, upsampling=4):
    ...
```

#### Use `DocumentArray` instead of Generator

```python
from tests.data_generator import fashion_match_documentarray as mda

from tests.data_generator import qa_match_documentarray as mda

da = mda()  # slow, as it scans over all data
```


    
