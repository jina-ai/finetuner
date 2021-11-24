# Finetuning a Transformer for Question-Answering

```{tip}
This example is inspired by [`jina hello chatbot`](https://docs.jina.ai/get-started/hello-world/covid-19-chatbot/). We stronly recommend you to checkout that demo first before go through this tutorial.
```

````{info}
This example will only with the `pytorch` framework, as it requires the use of `transformers` package. You can install the package by using

```
pip install transformers
```
````

In this example, we want to "tune" a transformer model on Covid19 QA data, the same dataset that we are using in `jina hello chatbot`.

Precisely, "tuning" means: 
- we set up a Jina search pipeline and will look at the top-K semantically similar questions;
- we accept or reject the results based on their quality;
- we let the model remember our feedback and produce better search results.

Hopefully the procedure converges after several rounds and we get a tuned embedding for better search tasks.

## Build embedding model

Let's create a transformers model as our {ref}`embedding model<embedding-model>`, where we will use the `[CLS]` token as the embedding:


```python
import torch
from transformers import AutoModel

TRANSFORMER_MODEL = 'sentence-transformers/paraphrase-MiniLM-L6-v2'


class TransformerEmbedder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained(TRANSFORMER_MODEL)

    def forward(self, inputs):
        out_model = self.model(**inputs)
        cls_token = out_model.last_hidden_state[:, 0, :]
        return cls_token
```
## Prepare data

Now prepare CovidQA data for the Finetuner. Note that Finetuner accepts Jina `DocumentArray`/`DocumentArrayMemmap`, so we first convert the data into this format.

```python
from finetuner.toydata import generate_qa
```

`generate_qa` is a generator that yields every question as a `Document` object.

```bash
<jina.types.document.Document id=dc315d50-1bae-11ec-b32d-1e008a366d49 tags={'wrong_answer': "If you have been in...', 'answer': 'Your doctor ...'} at 5794172560>
```

Now, here each `Document` will have only a `text` attribute, but, as we know, transformers need tokens as inputs. So we need to transform the inputs into tokens before feeding them to the model.

The best place to do this is when "collating" all the items together into a batch. This enables us to dynamically pad the batch to the length of the largest example in the batch, and not the maximum length allowed in the model - which makes the model run faster and consume less memory.

We can do this by constructing a collation function that we will later pass to the `fit` function

```python
from typing import List
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL)

def collate_fn(inputs: List[str]):
    batch_tokens = tokenizer(
        inputs,
        truncation=True,
        max_length=50,
        padding=True,
        return_tensors='pt',
    )
    return batch_tokens
```

## Put together

Finally, let's feed the model and the data into the Finetuner:

```python
import finetuner

finetuner.fit(
   TransformerEmbedder(),
   train_data=generate_qa(),
   collate_fn=collate_fn,
   interactive=True)
```

## Label interactively

From the left bar, select `.text` as the Field. You can now label the data by mouse/keyboard. The model will get trained and improved as you are labeling.

```{figure} covid-labeler.gif
:align: center
```
