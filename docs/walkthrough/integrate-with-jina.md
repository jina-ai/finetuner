# Integration

(integrate-with-jina)=
## Fine-tuned model as Executor

Once fine-tuning is finished, it's time to actually use the model. 
Finetuner, being part of the Jina ecosystem, provides a convenient way to use tuned models via [Jina Executors](https://docs.jina.ai/fundamentals/executor/).

We've created the [`FinetunerExecutor`](https://hub.jina.ai/executor/13dzxycc) which can be added in a [Jina Flow](https://docs.jina.ai/fundamentals/flow/) and load any tuned model. 
More specifically, the executor exposes an `/encode` endpoint that embeds [Documents](https://docarray.jina.ai/fundamentals/document/) using the fine-tuned model.

Loading a tuned model is simple! You just need to provide a few parameters under the `uses_with` argument when adding the `FinetunerExecutor` to the [Flow]((https://docs.jina.ai/fundamentals/flow/)).
You have three options:

````{tab} Artifact id and token
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
    uses='jinahub+docker://FinetunerExecutor/v0.9.2',  # use v0.9.2-gpu for gpu executor.
    uses_with={'artifact': run.artifact_id, 'token': token},
)
```
````
````{tab} Locally saved artifact
```python
from jina import Flow
	
f = Flow().add(
    uses='jinahub+docker://FinetunerExecutor/v0.9.2',  # use v0.9.2-gpu for gpu executor.
    uses_with={'artifact': '/mnt/YOUR-MODEL.zip'},
    volumes=['/your/local/path/:/mnt']  # mount your model path to docker.
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
  uses: jinahub+docker://FinetunerExecutor/v0.9.2
  with:
    artifact: 'COPY-YOUR-ARTIFACT-ID-HERE'
    token: 'COPY-YOUR-TOKEN-HERE'  # or better set as env
```
````

As you can see, it's super easy! 
If you did not call `save_artifact`,
you need to provide the `artifact_id` and `token`.
`FinetunerExecutor` will automatically pull your model from the cloud storage to the container.

On the other hand,
if you have saved artifact locally,
please mount the zipped artifact to the docker container.
`FinetunerExecutor` will unzip the artifact and load models.

You can start your flow with:

```python
with f:
    # in this example, we fine-tuned a BERT model and embed a Document..
    returned_docs = f.post(
        on='/encode',
        inputs=DocumentArray(
            [
                Document(
                    text='some text to encode'
                )
            ]
        )
    )

for doc in returned_docs:
    print(f'Text of the returned document: {doc.text}')
    print(f'Shape of the embedding: {doc.embedding.shape}')
```

```console
Text of the returned document: some text to encode
Shape of the embedding: (768,)
```

In order to see what other options you can specify when initializing the executor, please go to the [`FinetunerExecutor`](https://hub.jina.ai/executor/13dzxycc) page and click on `Arguments` on the top-right side.

```{admonition} FinetunerExecutor parameters
:class: tip
The only required argument is `artifact`. We provide default values for others.
```

(integrate-with-docarray)=
## Embed DocumentArray

Similarly, you can embed the [DocumentArray](https://docarray.jina.ai/) with fine-tuned model:

````{tab} Artifact id and token
```python
from docarray import DocumentArray, Document
import finetuner

finetuner.login()

token = finetuner.get_token()
run = finetuner.get_run(
    experiment_name='YOUR-EXPERIMENT',
    run_name='YOUR-RUN'
)

da = DocumentArray([Document(text='some text to encode')])

da.post(
    'jinahub+docker://FinetunerExecutor/v0.9.2/encode',
    uses_with={'artifact': run.artifact_id, 'token': token},
)
```
````
````{tab} Locally saved artifact
```python
from docarray import DocumentArray, Document

da = DocumentArray([Document(text='some text to encode')])

da.post(
    'jinahub+docker://FinetunerExecutor/v0.9.2/encode,
    uses_with={'artifact': '/mnt/YOUR-MODEL.zip'},
    volumes=['/your/local/path/:/mnt']  # mount your model path to docker.
)
```
````

```console
Text of the returned document: some text to encode
Shape of the embedding: (768,)
```
