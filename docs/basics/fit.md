# One-liner `fit()`

```{include} ../index.md
:start-after: <!-- start fit-method -->
:end-before: <!-- end fit-method -->
```

Note, `finetuner.fit` always returns the tuned model. You can save the model via `finetuner.save(model)`.

## Save model

```python
import finetuner

finetuner.save(model, './saved-model')
```

```{caution}
Depending on your framework, `save` can behave differently. Tensorflow/Keras saves model architecture with the parameters, whereas PyTorch & Paddle only saves the trained parameters.
```

## Display model

```python
import finetuner

finetuner.display(model)
```

```{caution}
Depending on your framework, `display` may require different argument for rendering the model correctly. In PyTorch & Paddle, you will also need to give the `input_size` and sometimes `input_dtype` to correctly render the model.
```

More information can be {ref}`found here<display-method>`.

## Embed documents

You can use `finetuner.embed()` method to compute the embeddings of a `DocumentArray` or `DocumentArrayMemmap`.

```python
import finetuner
from jina import DocumentArray

docs = DocumentArray(...)

finetuner.embed(docs, model)

print(docs.embeddings)
```

Note that, `model` above must be an {term}`Embedding model`.



## Example

```python
import finetuner
import torch

from collections import Counter
from finetuner.toydata import generate_qa
from jina import Document
from typing import List

VOCAB_SIZE = 5000
PAD_TOKEN = '<pad>'
PAD_INDEX = 0
UNK_TOKEN = '<unk>'
UNK_INDEX = 1

train_data = generate_qa(num_neg=1)
tokens = [token for doc in train_data for token in doc.text.split()]
vocab = {
    key: idx + 2
    for idx, (key, _) in enumerate(Counter(tokens).most_common(VOCAB_SIZE - 2))
}
vocab.update({PAD_TOKEN: PAD_INDEX, UNK_TOKEN: UNK_INDEX})

class LastCell(torch.nn.Module):
    def forward(self, x):
        out, _ = x
        return out[:, -1, :]

embed_model = torch.nn.Sequential(
    torch.nn.Embedding(num_embeddings=VOCAB_SIZE, embedding_dim=64),
    torch.nn.LSTM(64, 64, bidirectional=True, batch_first=True),
    LastCell(),
    torch.nn.Linear(in_features=2 * 64, out_features=32)
)

def preprocess_fn(doc: Document) -> torch.LongTensor:
    return torch.LongTensor([vocab.get(token, UNK_INDEX) for token in doc.text.split()])


def collate_fn(tensors: List[torch.LongTensor]) -> torch.LongTensor:
    return torch.nn.utils.rnn.pad_sequence(tensors, padding_value=PAD_INDEX, batch_first=True).long()

model = finetuner.fit(
    embed_model,
    train_data=train_data,
    eval_data=generate_qa(num_neg=1),
    epochs=1,
    preprocess_fn=preprocess_fn,
    collate_fn=collate_fn,
)

finetuner.display(model, input_size=(100,), input_dtype='long')
```

```console
  Training [1/1] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 6/6 0:00:00 0:00:57 • loss: 0.664
                                                                    
  name          output_shape_display         nb_params   trainable  
 ────────────────────────────────────────────────────────────────── 
  embedding_1   [100, 64]                    320000      True       
  lstm_2        [[[2, 1, 64], [2, 1, 64]]]   66560       False      
  lastcell_3    [128]                        0           False      
  linear_4      [32]                         4128        True       
                                                                    
Green layers are trainable layers, Cyan layers are non-trainable layers or frozen layers.
Gray layers indicates this layer has been replaced by an Identity layer.
Use to_embedding_model(...) to create embedding model.
```

```python
finetuner.save(model, './saved-model')
```

```python
from jina import DocumentArray

all_q = DocumentArray(generate_qa())
finetuner.embed(all_q, model, preprocess_fn=preprocess_fn, collate_fn=collate_fn)
print(all_q.embeddings.shape)
```

```console
(481, 32)
```

```python
all_q.plot_embeddings()
```

```{figure} embed.png
:align: center
:width: 80%
```