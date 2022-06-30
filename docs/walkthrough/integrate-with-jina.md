(integrate-with-jina)=
# Integrate with Jina

Once fine-tuning is finished, it's time to actually use the model. 
Finetuner, being part of the Jina ecosystem, provides a convenient way to use tuned models via [Jina Executors](https://docs.jina.ai/fundamentals/executor/).

We've created the [`FinetunerExecutor`](https://hub.jina.ai/executor/13dzxycc) which can be added in a [Jina Flow](https://docs.jina.ai/fundamentals/flow/) and load any tuned model. 

Loading a tuned model is simple! You just need to provide a few parameters under the `uses_with` argument when adding the `FinetunerExecutor` to the [Flow]((https://docs.jina.ai/fundamentals/flow/)).

````{tab} via Docker image (recommended)
```python
from jina import Flow
	
f = Flow().add(
    uses='jinahub+docker://FinetunerExecutor',
    uses_with={'artifact': 'model_dir/tuned_model', 'batch_size': 16},
)
```
````
````{tab} via source code
```python
from jina import Flow
	
f = Flow().add(
    uses='jinahub://FinetunerExecutor',
    uses_with={'artifact': 'model_dir/tuned_model', 'batch_size': 16},
)
```
````
As you can see, it's super easy! We just provided the model path and the batch size.

In order to see what other options you can specify when initializing the executor, please go to the [`FinetunerExecutor`](https://hub.jina.ai/executor/13dzxycc) page and click to the `Arguments` button on the top-right side.

```{admonition} FinetunerExecutor parameters
:class: tip
The only required argument is `artifact`. We provide default values for others.
```


## Using the `FinetunerExecutor`

Here's a simple code snippet demonstrating the `FinetunerExecutor` usage in the flow:

```python
from docarray import DocumentArray, Document
from jina import Flow

f = Flow().add(
    uses='jinahub+docker://FinetunerExecutor',
    uses_with={'artifact': 'model_dir/tuned_model', 'batch_size': 16},
)

with f:
    returned_docs = f.post(on='/', inputs=DocumentArray([Document()]))

for doc in returned_docs:
    print(f'Document returned with text: "{doc.text}"')
    print(f'Shape of the embedding: {doc.embedding.shape}')
```