# One-liner `fit()`

```{include} ../index.md
:start-after: <!-- start fit-method -->
:end-before: <!-- end fit-method -->
```

Note, `finetuner.fit` always returns two objects in a tuple: the tuned model and a summary statistic of the fitting procedure. You can save the model via `finetuner.save(model)`. You can save the summary object by calling `.save()`; or plot it via `.plot()` and see how the loss on training & evaluation data changes over time.

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
import torch
from finetuner.toydata import generate_qa_match
import finetuner

class LastCell(torch.nn.Module):
    def forward(self, x):
        out, _ = x
        return out[:, -1, :]

embed_model = torch.nn.Sequential(
    torch.nn.Embedding(num_embeddings=5000, embedding_dim=64),
    torch.nn.LSTM(64, 64, bidirectional=True, batch_first=True),
    LastCell(),
    torch.nn.Linear(in_features=2 * 64, out_features=32))

model, summary = finetuner.fit(
    embed_model,
    train_data=lambda: generate_qa_match(num_neg=5, max_seq_len=10),
    eval_data=lambda: generate_qa_match(num_neg=5, max_seq_len=10),
    epochs=3,
)

finetuner.display(model, input_size=(100,), input_dtype='long')
```

```console
⠼  Epoch 1/3 ━╸━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 0:00:00 estimating... 
⠼       DONE ━━━━━━━━━━━━━╸━━━━━━━━━━━━━━━━━━━━━━━━━━━ 0:00:01 10.2 step/s train: 1.25 | Eval Loss: 1.06
⠇       DONE ━━━╸━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 0:00:01 100% ETA: 0 seconds train: 1.14 | Eval Loss: 1.02
⠹       DONE ━━━╸━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 0:00:01 100% ETA: 0 seconds train: 1.10 | Eval Loss: 0.98
                                                                    
  name          output_shape_display         nb_params   trainable  
 ────────────────────────────────────────────────────────────────── 
  embedding_1   [100, 64]                    320000      True       
  lstm_2        [[[2, 2, 64], [2, 2, 64]]]   66560       False      
  lastcell_3    [128]                        0           False      
  linear_4      [32]                         4128        True       
                                                                    
Green layers can be used as embedding layers, whose name can be used as 
layer_name in to_embedding_model(...).
```

```python
finetuner.save(model, './saved-model')
summary.plot('fit.png')
```

```{figure} fit-plot.png
:align: center
:width: 80%
```

```python
from jina import DocumentArray
all_q = DocumentArray(generate_qa_match())
finetuner.embed(all_q, model)
print(all_q.embeddings.shape)
```

```console
(481, 32)
```

```python
all_q.visualize('embed.png', method='tsne')
```

```{figure} embed.png
:align: center
:width: 80%
```