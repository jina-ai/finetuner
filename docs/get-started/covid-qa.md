# Finetuning Bi-LSTM for Question-Answering

```{tip}
This example is inspired by [`jina hello chatbot`](https://docs.jina.ai/get-started/hello-world/covid-19-chatbot/). We stronly recommend you to checkout that demo first before go through this tutorial.
```


In this example, we want to "tune" the 32-dim embedding vectors from a bidirectional LSTM on Covid19 QA data, the same dataset that we are using in `jina hello chatbot`. 

Precisely, "tuning" means: 
- we set up a Jina search pipeline and will look at the top-K semantically similar questions;
- we accept or reject the results based on their quality;
- we let the model to remember our feedback and produces better search result.

Hopefully the procedure converges after several rounds; and we get a tuned embedding for better search task.

## Build embedding model

Let's write a 2-layer MLP as our {ref}`embedding model<embedding-model>` using any of the following framework.

````{tab} PyTorch

```python
import torch


class LastCell(torch.nn.Module):
    def forward(self, x):
        out, _ = x
        return out[:, -1, :]


embed_model = torch.nn.Sequential(
    torch.nn.Embedding(num_embeddings=5000, embedding_dim=64),
    torch.nn.LSTM(64, 64, bidirectional=True, batch_first=True),
    LastCell(),
    torch.nn.Linear(in_features=2 * 64, out_features=32))
```

````
````{tab} Keras
```python
import tensorflow as tf

embed_model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=5000, output_dim=64),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(32)])
```
````
````{tab} Paddle
```python
import paddle


class LastCell(paddle.nn.Layer):
    def forward(self, x):
        out, _ = x
        return out[:, -1, :]


embed_model = paddle.nn.Sequential(
    paddle.nn.Embedding(num_embeddings=5000, embedding_dim=64),
    paddle.nn.LSTM(64, 64, direction='bidirectional'),
    LastCell(),
    paddle.nn.Linear(in_features=2 * 64, out_features=32))
```
````

## Prepare data

Now prepare CovidQA data for the Finetuner. Note that Finetuner accepts Jina `DocumentArray`/`DocumentArrayMemmap`, so we first convert them into this format.

```python
from finetuner.toydata import generate_qa_match
```

`generate_qa_match` is a generator that yields every question as a `Document` object. 
It also codes and pads the question into a 100-dimensional array, which is stored in `blob`.

```bash
<jina.types.document.Document id=dc315d50-1bae-11ec-b32d-1e008a366d49 tags={'wrong_answer': "If you have been in...', 'answer': 'Your doctor ...'} blob={'dense': {'buffer': 'AAAAAAAAAAAAAAAA...', 'shape': [100], 'dtype': '<i8'}} at 5794172560>
```

## Put together

Finally, let's feed the model and the data into the Finetuner:

```python
import finetuner

finetuner.fit(
   embed_model,
   train_data=generate_qa_match,
   interactive=True)
```

## Label interactively

From the left bar, select `text` as the view.

In the content, select `.tags` and then fill in `question` to tell the UI renders text from `Document.tags['question']`. 

You can now label the data by mouse/keyboard. The model will get trained and improved as you are labeling.

```{figure} covid-labeler.gif
:align: center
```
