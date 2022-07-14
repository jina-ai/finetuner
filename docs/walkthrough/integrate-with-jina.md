# Integration

(integrate-with-docarray)=
## Embed your DocumentArray

placeholder


(integrate-with-jina)=
## Integrate with Jina

Once fine-tuning is finished, it's time to actually use the model. 
Finetuner, being part of the Jina ecosystem, provides a convenient way to use tuned models via [Jina Executors](https://docs.jina.ai/fundamentals/executor/).

We've created the [`FinetunerExecutor`](https://hub.jina.ai/executor/13dzxycc) which can be added in a [Jina Flow](https://docs.jina.ai/fundamentals/flow/) and load any tuned model. 
More specifically, the executor exposes an `/encode` endpoint that embeds [Documents](https://docarray.jina.ai/fundamentals/document/) using the fine-tuned model.

Loading a tuned model is simple! You just need to provide a few parameters under the `uses_with` argument when adding the `FinetunerExecutor` to the [Flow]((https://docs.jina.ai/fundamentals/flow/)).

````{tab} Python
```python
import finetuner
from jina import Flow

finetuner.login()

token = finetuner.get_token()
run = finetuner.get_run(
    experiment_name='YOUR-EXPERIMENT',
    run_name='YOUR-RUN'
)
	
f = Flow().add(
    uses='jinahub+docker://FinetunerExecutor/v0.9.1',  # use v0.9.1-gpu for gpu executor.
    uses_with={'artifact': run.artifact_id, 'token': token},
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
  uses: jinahub+docker://FinetunerExecutor/v0.9.1
  with:
    artifact: 'COPY-YOUR-ARTIFACT-ID-HERE'
    token:'COPY-YOUR-TOKEN-HERE'  # or better set as env
```
````
```{admonition} FinetunerExecutor via source code
:class: tip
You can also use the `FinetunerExecutor` via source code by specifying `jinahub://FinetunerExecutor` under the `uses` parameter.
However, using docker images is recommended.
```

Then you can start your flow with:

```python
with f:
    # in this example, we fine-tuned a BERT model and embed a Document with some random text.
    returned_docs = f.post(on='/encode', inputs=DocumentArray([Document(text='some text to encode')]))

for doc in returned_docs:
    print(f'Text of the returned document: {doc.text}')
    print(f'Shape of the embedding: {doc.embedding.shape}')
```

```console
Text of the returned document: hello
Shape of the embedding: (1, 768)
```

As you can see, it's super easy! We just provided the `artifact_id` and your `token`.

In order to see what other options you can specify when initializing the executor, please go to the [`FinetunerExecutor`](https://hub.jina.ai/executor/13dzxycc) page and click on `Arguments` on the top-right side.

```{admonition} FinetunerExecutor parameters
:class: tip
The only required argument is `artifact`. We provide default values for others.
```
