# Inference

Once fine-tuning is finished, it's time to actually use the model.
You can use the fine-tuned models directly to encode DocumentArray objects or to set up an encoding service.
When encoding, data can also be provided as a regular list.

```{admonition} Use FinetunerExecutor inside a Jina Flow
:class: hint
Finetuner offers the {class}`~finetuner.encode` interface to embed your data locally
If you would like to use fine-tuned model inside a Jina Flow as an Executor, checkout
{doc}`/advanced-topics/finetuner-executor`.
```

(integrate-with-list)=
## Encoding a List
Data that is stored in a regular list can be embedded in the same way you would embed a DocumentArray.
Since the modality of your input data can be inferred from the model being used, there is no need to provide any additional information besides the content you want to encode.
When providing data as a list, the `finetuner.encode` method will return a `np.ndarray` of embeddings, instead of a `docarray.DocumentArray`:

````{tab} Artifact id and token
```python
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
)

texts = ['some text to encode']

embeddings = finetuner.encode(model=model, data=texts)

for text, embedding in zip(texts, embeddings):
    print(f'Text of the returned document: {text}')
    print(f'Shape of the embedding: {embedding.shape}')
```
````
````{tab} Locally saved artifact
```python
import finetuner

model = finetuner.get_model('/path/to/YOUR-MODEL.zip')

texts = ['some text to encode']

embeddings = finetuner.encode(model=model, data=texts)

for text, embedding in zip(texts, embeddings):
    print(f'Text of the returned document: {text}')
    print(f'Shape of the embedding: {embedding.shape}')
```
````
````{tab} (Special case) CLIP inference
```python
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
    select_model='clip-text'  # use `clip-vision` to encode image.
)

texts = ['some text to encode']
embeddings = finetuner.encode(model=model, data=texts)

for text, embedding in zip(texts, embeddings):
    print(f'Text of the returned document: {text}')
    print(f'Shape of the embedding: {embedding.shape}')
```
````


```{admonition} Inference with ONNX
:class: tip
In case you set `to_onnx=True` when calling `finetuner.fit` function,
please use `model = finetuner.get_model('/path/to/YOUR-MODEL.zip', is_onnx=True)`.
```

```{admonition} Encoding other Modalities
:class: tip
Of course you can not only encode texts.
For encoding a list of images, you can provide URIs, e.g.,
`embeddings = finetuner.encode(model=model, data=['path/to/apple.png'])`.
```

(integrate-with-docarray)=
## Encoding a DocumentArray

To embed a DocumentArray with a fine-tuned model, you can get the model of your Run via the {func}`~finetuner.get_model` function and embed it via the {func}`finetuner.encode` function:

````{tab} Artifact id and token
```python
from finetuner import DocumentArray, Document
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
from finetuner import DocumentArray, Document
import finetuner

model = finetuner.get_model('/path/to/YOUR-MODEL.zip')

da = DocumentArray([Document(text='some text to encode')])
finetuner.encode(model=model, data=da)

for doc in da:
    print(f'Text of the returned document: {doc.text}')
    print(f'Shape of the embedding: {doc.embedding.shape}')
```
````
````{tab} (Special case) CLIP inference
```python
from finetuner import DocumentArray, Document
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
    select_model='clip-text'  # use `clip-vision` to encode image.
)

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
