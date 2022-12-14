# Encode Documents

Once fine-tuning is finished, it's time to actually use the model.
You can use the fine-tuned models directly to encode [DocumentArray](https://docarray.jina.ai/) objects or setting up an encoding service.
When encoding, data can also be provided as a regular list.

(integrate-with-docarray)=
## Embed DocumentArray

To embed a [DocumentArray](https://docarray.jina.ai/) with a fine-tuned model, you can get the model of your Run via the {func}`~finetuner.get_model` function and embed it via the {func}`finetuner.encode` function:

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

model = finetuner.get_model(
    run.artifact_id,
    token=token,
    device='cuda', # model will be placed on cpu by default.
)

da = DocumentArray([Document(text='some text to encode')])

finetuner.encode(model=model, data=da)

for doc in da:
    print(f'Text of the returned document: {doc.text}')
    print(f'Shape of the embedding: {doc.embedding.shape}')
```
````
````{tab} Locally saved artifact
```python
from docarray import DocumentArray, Document
import finetuner

model = finetuner.get_model('/path/to/YOUR-MODEL.zip')

da = DocumentArray([Document(text='some text to encode')])

finetuner.encode(model=model, data=da)

for doc in da:
    print(f'Text of the returned document: {doc.text}')
    print(f'Shape of the embedding: {doc.embedding.shape}')
```
````

```console
Text of the returned document: some text to encode
Shape of the embedding: (768,)
```

## Encoding a List
Data that is stored in a regular list can be embedded in the same way you would a [DocumentArray](https://docarray.jina.ai/). Since the modality of your input data can be inferred from the model being used, there is no need to provide any additional information besides the content you want to encode. When providing data as a list, the `finetuner.encode` method will return a `np.ndarray` of embeddings, instead of a `docarray.DocumentArray`:

```python
from docarray import DocumentArray, Document
import finetuner

model = finetuner.get_model('/path/to/YOUR-MODEL.zip')

texts = ['some text to encode']

embeddings = finetuner.encode(model=model, data=texts)

for text, embedding in zip(texts, embeddings):
    print(f'Text of the returned document: {text}')
    print(f'Shape of the embedding: {embedding.shape}')
```


```{admonition} Inference with ONNX
:class: tip
In case you set `to_onnx=True` when calling `finetuner.fit` function,
please use `model = finetuner.get_model('/path/to/YOUR-MODEL.zip', is_onnx=True)`
```

(integrate-with-jina)=
## Fine-tuned model as Executor

Finetuner, being part of the Jina AI Cloud, provides a convenient way to use tuned models via [Jina Executors](https://docs.jina.ai/fundamentals/executor/).

We've created the [`FinetunerExecutor`](https://cloud.jina.ai/executor/13dzxycc) which can be added in a [Jina Flow](https://docs.jina.ai/fundamentals/flow/) and load any tuned model. 
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
    uses='jinahub+docker://FinetunerExecutor/latest',  # use latest-gpu for gpu executor.
    uses_with={'artifact': run.artifact_id, 'token': token},
)
```
````
````{tab} Locally saved artifact
```python
from jina import Flow
	
f = Flow().add(
    uses='jinahub+docker://FinetunerExecutor/latest',  # use latest-gpu for gpu executor.
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
  uses: jinahub+docker://FinetunerExecutor/latest
  with:
    artifact: 'COPY-YOUR-ARTIFACT-ID-HERE'
    token: 'COPY-YOUR-TOKEN-HERE'  # or better set as env
```
````

As you can see, it's super easy! 
If you did not call {func}`~finetuner.run.Run.save_artifact`,
you need to provide the `artifact_id` and `token`.
`FinetunerExecutor` will automatically pull your model from the Jina AI Cloud to the container.

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

In order to see what other options you can specify when initializing the executor, please go to the [`FinetunerExecutor`](https://cloud.jina.ai/executor/13dzxycc) page and click on `Arguments` on the top-right side.

```{admonition} FinetunerExecutor parameters
:class: tip
The only required argument is `artifact`. We provide default values for others.
```

## Special case: Artifacts with CLIP models 
If your fine-tuning job was executed on a CLIP model, your artifact contains two 
models: `clip-vision` and `clip-text`.
The vision model allows you to embed images and the text model can encode text passages
into the same vector space.
To use those models, you have to provide the name of the model via an additional
`select_model` parameter to the {func}`~finetuner.get_model` function.


````{tab} CLIP text model
```python
from docarray import DocumentArray, Document
import finetuner

finetuner.login()

token = finetuner.get_token()
run = finetuner.get_run(
    experiment_name='YOUR-EXPERIMENT',
    run_name='YOUR-RUN'
)

model = finetuner.get_model(
    run.artifact_id,
    token=token,
    device='cuda',
    select_model='clip-text'
)

da = DocumentArray([Document(text='some text to encode')])

finetuner.encode(model=model, data=da)
```
````
````{tab} CLIP vision model
```python
from docarray import DocumentArray, Document
import finetuner

finetuner.login()

token = finetuner.get_token()
run = finetuner.get_run(
    experiment_name='YOUR-EXPERIMENT',
    run_name='YOUR-RUN'
)

model = finetuner.get_model(
    run.artifact_id,
    token=token,
    device='cuda',
    select_model='clip-vision'
)

da = DocumentArray([Document(text='~/Pictures/my_img.png')])

finetuner.encode(model=model, data=da)
```
````

If you want to host the CLIP models, you also have to provide the name of the model via the
`select_model` parameter inside the `uses_with` attribute:

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
    uses='jinahub+docker://FinetunerExecutor/latest',  # use latest-gpu for gpu executor.
    uses_with={
        'artifact': run.artifact_id, 'token': token, 'select_model': 'clip-vision'
    },
)

```