# trainer

## Dev Install

Trainer requires your local `jina` as the latest master. It is the best if you have `jina` installed via `pip install -e .`.

```bash
git clone https://github.com/jina-ai/trainer.git
cd trainer
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

## Synthetic Matching Data

We use Fashion-MNIST and Covid QA data for generating synthetic matching data, as these two datasets align with the first two `jina hello` demos.

- `tests.data_generator.fashion_match_doc_generator()`: the generator of Fashion-MNIST synthetic matching data.
- `tests.data_generator.qa_match_doc_generator()`: the generator of Fashion-MNIST synthetic matching data.

### Fashion-MNIST as synthetic matching data

Fashion-MNIST contains 60,000 training images and 10,000 images in 10 classes. Each image is a single channel 28x28 grayscale image. To convert this dataset into match data, we build each document to contain the following info that are relevant to `trainer`:

  - `.blob`: the image
  - `.matches`: the generated positive & negative matches Document
    - `.blob`: the matched Document's image 
    - `.tags['trainer']['label']`: the match label, can be `1` or `-1` or user-defined.

Matches are built with the logic below:
    - sample same-class documents as positive matches;
    - sample other-class documents as negative matches.

### Covid QA as synthetic matching data

Covid QA data is a CSV that has 481 rows with columns `question`, `answer` & `wrong_answer`. To convert this dataset into match data, we build each document to contain the following info that are relevant to `trainer`:

 - `.text`: the original `question` column
 - `.blob`: when set `to_ndarray` to True, `text` is tokenized into a fixed length `ndarray`.
 - `.matches`: the generated positive & negative matches Document
   - `.text`: the original `answer`/`wrong_answer` column
   - `.tags['trainer']['label']`: the match label, can be `1` or `-1` or user-defined.
   - `.blob`: when set `to_ndarray` to True, `text` is tokenized into a fixed length `ndarray`.

Matches are built with the logic below:
    - only allows 1 positive match per Document, it is taken from the `answer` column.
    - always include `wrong_answer` column as the negative match. Then sample other documents' answer as negative matches.


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


## Example 1: train arbitrary DNN for `jina hello fashion`

1. Use artificial pairwise data to train `user_model` in a siamese manner:

   <details>
   <summary>Using KerasTrainer</summary>

   - build a simple dense network with bottleneck

      ```python
     import tensorflow as tf

     user_model = tf.keras.Sequential(
         [
             tf.keras.layers.Flatten(input_shape=(28, 28)),
             tf.keras.layers.Dense(128, activation='relu'),
             tf.keras.layers.Dense(32),
         ]
     )
     ```

   - wrap the user model with our trainer
      ```python
      from trainer.keras import KerasTrainer

      kt = KerasTrainer(user_model, head_layer='CosineLayer')
      ```

   - fit and save the checkpoint

      ```python
      from tests.data_generator import fashion_match_doc_generator as fmdg

      kt.fit(fmdg, epochs=1)
      kt.save('./examples/fashion/trained')
      ```

   </details>

   <details>
   <summary>Using PytorchTrainer</summary>

   - build a simple dense network with bottleneck:
       ```python
       import torch.nn as nn

       user_model = nn.Sequential(
           nn.Flatten(),
           nn.Linear(in_features=784, out_features=128),
           nn.ReLU(),
           nn.Linear(in_features=128, out_features=10)
       )
       ```

   - wrap the user model with our trainer:
       ```python
       from trainer.pytorch import PytorchTrainer

       pt = PytorchTrainer(user_model, head_layer='CosineLayer')
       ```

   - fit and save the checkpoint:

       ```python
       from tests.data_generator import fashion_match_documentarray as fmdg

       pt.fit(fmdg(num_total=50), epochs=10)
       pt.save('./examples/fashion/trained.pt')
       ```

   </details>

   <details>
   <summary>Using PaddleTrainer</summary>
   
    - build a simple dense network with bottleneck:
   
        ```python
        from paddle import nn
        user_model = nn.Sequential(
            nn.Flatten(start_axis=1),
            nn.Linear(in_features=784, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=32)
        )
        ```
    - wrap the user model with our trainer
   
        ```python
       from trainer.paddle import PaddleTrainer
      
       pt = PaddleTrainer(user_model, head_layer='CosineLayer')
       ```
      
    - fit and save the checkpoint
   
        ```python
       from tests.data_generator import fashion_match_documentarray as fmdg

       pt.fit(fmdg(num_total=50), epochs=10)
       
       from paddle.static import InputSpec
       x_spec = InputSpec(shape=[None, 28, 28], name='x')
       pt.save('examples/fashion/paddle_ckpt', input_spec=[x_spec])
       ```
   </details>

2. Observe the decreasing of training loss and increasing of the accuracy.

3. Test `trained` model in the Jina `hello fashion` pipeline:
    ```bash
    python examples/fashion/app.py
    ```

4. Open the browser and check the results.

    
