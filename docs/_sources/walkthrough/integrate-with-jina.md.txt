(integrate-with-jina)=
# Integrate with Jina

Once fine-tuning is finished, it's time to actually use the model. 
Finetuner, being part of the Jina ecosystem, provides a convenient way to use tuned models via [Jina Executors](https://docs.jina.ai/fundamentals/executor/).

We've created the [`FinetunerExecutor`](https://hub.jina.ai/executor/13dzxycc) which can be added in a [Jina Flow](https://docs.jina.ai/fundamentals/flow/) and load any tuned model. 
More specifically, the executor exposes an `/encode` endpoint that embeds [Documents](https://docarray.jina.ai/fundamentals/document/) using the fine-tuned model.

Loading a tuned model is simple! You just need to provide a few parameters under the `uses_with` argument when adding the `FinetunerExecutor` to the [Flow]((https://docs.jina.ai/fundamentals/flow/)).

````{tab} Python
```python
from jina import Flow
	
f = Flow().add(
    uses='jinahub+docker://FinetunerExecutor',
    uses_with={'artifact': 'model_dir/tuned_model', 'batch_size': 16},
)
```
````
````{tab} YAML
```yaml
jtype: Flow
with:
  port: 51000
  protocol: grpc
executors:
  uses: jinahub+docker://FinetunerExecutor
  with:
    artifact: 'model_dir/tuned_model'
    batch_size: 16
```
````
```{admonition} FinetunerExecutor via source code
:class: tip
You can also use the `FinetunerExecutor` via source code by specifying `jinahub://FinetunerExecutor` under the `uses` parameter.
However, using docker images is recommended.
```

As you can see, it's super easy! We just provided the model path and the batch size.

In order to see what other options you can specify when initializing the executor, please go to the [`FinetunerExecutor`](https://hub.jina.ai/executor/13dzxycc) page and click on `Arguments` on the top-right side.

```{admonition} FinetunerExecutor parameters
:class: tip
The only required argument is `artifact`. We provide default values for others.
```


## Using `FinetunerExecutor`

Here's a simple code snippet demonstrating the `FinetunerExecutor` usage in the Flow:

```python
from docarray import DocumentArray, Document
from jina import Flow

f = Flow().add(
    uses='jinahub+docker://FinetunerExecutor',
    uses_with={'artifact': 'model_dir/tuned_model', 'batch_size': 16},
)

with f:
    returned_docs = f.post(on='/encode', inputs=DocumentArray([Document(text='hello')]))

for doc in returned_docs:
    print(f'Text of the returned document: {doc.text}')
    print(f'Shape of the embedding: {doc.embedding.shape}')
```
```bash
Text of the returned document: hello
Shape of the embedding: (1, 768)
```