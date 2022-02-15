# Using ONNX to deploy a finetuned model

Imagine a scenario where you successfully used Finetuner to finetune a model on your search problem using data from the respective domain. What comes next?

Naturally, you would want to deploy your {term}`embedding model` in a service and use it to
encode data as part of a bigger search application. [Jina](https://docs.jina.ai/)
provides the infrastructure layer to help you build and deploy neural search components.
By implementing custom `Executor` or using existing ones from [Jina Hub](https://hub.jina.ai/)
you can define the building blocks of your search application and glue everything
together in a fully-fledged search pipeline.

To use your model with Jina, you would have to build a new `Executor` tailored to your model and upload it to Jina Hub. You can then use a Jina `Flow` to deploy your search
application locally or in the cloud using kubernetes.

It is difficult to provide a unified `Executor` that can support all models trained with Finetuner,
since Finetuner supports different computational frameworks and each model differs in terms
of pre-processing required, input type etc.

There is an alternative though, that can be useful for standardizing your deployment
procedure and unifying different models. Finetuner allows the user to use ONNX for converting
finetuned models to the ONNX format, whether these models have been trained with PyTorch,
TensorFlow or PaddlePaddle. The ONNX format is an open standard for defining
neural network architectures. You can find out more in the [ONNX webpage](https://onnx.ai/).

After converting a trained model to the ONNX format, you can use the
[ONNXEncoder](https://hub.jina.ai/executor/2cuinbko) from Jina Hub to deploy your {term}`embedding
model`. The encoder simply loads your {term}`embedding model` in ONNX format and uses the ONNX
runtime for inference. The `Document` tensors are fed to the ONNX model as input
and the output NumPy vectors are assigned to the `Document` as embeddings.

The `ONNXEncoder` takes care of the embedding part and the only thing left is to add a custom
`Executor` for pre-processing in case there is the need for one. Additionally, you gain
a boost in efficiency, since the conversion to ONNX already optimizes the model based on the platform/hardware device.

This tutorial will walk you through a simple model finetuning and how you can convert to
ONNX and use the `ONNXEncoder` to deploy a Jina `Flow`.

````{info}
Jina, onnx, torch and torchvision are required to run this example. You can install the packages using:

```
pip install 'jina>=2.7' onnx torch torchvision
```
````


## Exporting to ONNX

Let's fine-tune a ResNet18 on a customized CelebA dataset.
[Download the dataset](https://static.jina.ai/celeba/celeba-img.zip) and decompress it to
`'./img_align_celeba'`.

```python
import torchvision
from docarray import DocumentArray

import finetuner as ft

data = DocumentArray.from_files('img_align_celeba/*.jpg')


def preprocess(doc):
    return (
        doc.load_uri_to_image_tensor(224, 224)
        .set_image_tensor_normalization()
        .set_image_tensor_channel_axis(-1, 0)
    )


data.apply(preprocess)

resnet = torchvision.models.resnet18(pretrained=True)

tuned_model = ft.fit(
    model=resnet,
    train_data=data,
    loss='TripletLoss',
    epochs=20,
    batch_size=128,
    to_embedding_model=True,
    input_size=(3, 224, 224),
    layer_name='adaptiveavgpool2d_67',
    freeze=False,
)

```

We can now export the model to the ONNX format, using the `to_onnx` method provided
by Finetuner. The `onnx` package is required to use this functionality. Also, if
you are using TensorFlow or PaddlePaddle you will need to install additional packages
for the ONNX conversion. Specifically you will need: 
[tf2onnx](https://github.com/onnx/tensorflow-onnx) for TensorFlow and
[paddle2onnx](https://github.com/PaddlePaddle/Paddle2ONNX) for PaddlePaddle.

```bash
pip install onnx
pip install tf2onnx
pip install paddle2onnx
```

Back to the ONNX conversion:

```python
from finetuner.tuner.onnx import to_onnx

to_onnx(tuned_model, 'tuned-model.onnx', input_shape=[3, 224, 224], opset_version=13)
```

The ONNX exported model should be stored now at `tuned-model.onnx`.

A helper function for validating the ONNX export is also provided:
```python
from finetuner.tuner.onnx import validate_onnx_export

validate_onnx_export(tuned_model, 'tuned-model.onnx', input_shape=[3, 224, 224])
```

This function compares the outputs of the native model and its ONNX counterpart against
the same random input. If the outputs differ an assertion error is raised.

Now that we have our model exported and have verified that it has the same behavior as the
original, it's time to deploy it 🚀.


## Deploying using the `ONNXEncoder`

You have already finetuned your own model and transferred it to ONNX format. Let's start deploying in the Jina `Flow`. If you are not familiar with Jina Hub or Jina `Flow`, check this:
[Use Hub Executor](https://docs.jina.ai/advanced/hub/use-hub-executor/)
[Use Jina Flow](https://docs.jina.ai/fundamentals/flow/)

Here are the steps you can follow:

### Add `ONNXEncoder` to Jina `Flow`

We already have `ONNXEncoder` on Jina Hub, let's add `ONNXEncoder` to Jina `Flow`:

using docker image:

```python
from jina import Flow

f = Flow().add(uses='jinahub+docker://ONNXEncoder',
                uses_with={'model_path': 'tuned-model.onnx'})
```

or using source code:

```python
from jina import Flow

f = Flow().add(uses='jinahub://ONNXEncoder',
                uses_with={'model_path': 'tuned-model.onnx'})
```

`model_path` is the path of the ONNX model you exported.

### Complete the `Flow` by adding indexer and starting the service

```python
from typing import Optional

import numpy as np
from docarray import Document, DocumentArray

from jina import Executor, Flow, requests


class SimpleIndexer(Executor):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._da = DocumentArray()

    @requests(on='/index')
    def index(self, docs: DocumentArray, **kwargs):
        self._da.extend(docs)

    @requests(on='/search')
    def search(self, docs: DocumentArray, **kwargs):
        docs.match(self._da)
        

f = Flow(port_expose=12345, 
         protocol='http').add(uses='jinahub://ONNXEncoder', 
                              uses_with={'model_path': 'tuned-model.onnx'},
                              name='Encoder').add(uses=SimpleIndexer, 
                                                  name='Indexer')
        
with f:
    f.post('/index', data, show_progress=True)
    f.block()
```

You will see something like this:

```bash
    Flow@6260[I]:🎉 Flow is ready to use!                                           
    🔗 Protocol:            HTTP
    🏠 Local access:        0.0.0.0:12345
    🔒 Private network:     192.168.31.213:12345
    💬 Swagger UI:          http://localhost:12345/docs
    📚 Redoc:               http://localhost:12345/redoc
⠼       DONE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╸ 0:00:14 100% ETA: 0 seconds 80 steps done in 14 seconds
```

which means that the service has been successfully started.

### Start to query!

Finally it's time to see what we can get from our service. Start a client and try to query using the service we have just built. 

```python
from docarray import Document, DocumentArray

from jina import Client
from jina.types.request.data import Response

# the callback function invoked when task is done
def print_matches(resp: Response):
   
    # print top-3 matches for first doc
    for idx, d in enumerate(resp.docs[0].matches[:3]):
        print(f'[{idx}]{d.scores["cosine"].value:2f}')


data = DocumentArray.from_files('img_align_celeba/*.jpg')

def preprocess(doc):
    return (
        doc.load_uri_to_image_tensor(224, 224)
        .set_image_tensor_normalization()
        .set_image_tensor_channel_axis(-1, 0)
    )


data.apply(preprocess)

# connect to localhost:12345
c = Client(protocol='http', port=12345)
c.post('/search', inputs=data, on_done=print_matches)
```

And outputs will be like this:

```bash
[0]0.000001
[1]0.080417
[2]0.115125
```

The first column is the index of images and the second column is the cosine distance.

Congratulations! You have implemented the pipeline which includes finetuning the model, converting to ONNX and deploying in Jina `Flow`.