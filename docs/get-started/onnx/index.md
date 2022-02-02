# Using ONNX to deploy a finetuned model

Imagine a scenario where you succesfully used finetuner to finetune a model on your
specific search problem using data from the respective domain. What comes next?

Naturally you would want to deploy your embedding model in a service and use it to
encode data as part of a bigger search application. Jina's [core framework](https://docs.jina.ai/)
provides the infrastructure layer to help you build and deploy neural search components.
By implementing custom executors or using existing ones from [Jina Hub](https://hub.jina.ai/)
you can define the building blocks of your search application and glue everything
together in a fully-fledged search pipeline.

In order to use your model with Jina, you would have to build a new executor tailored to your model
specifically and upload it to the hub. You can then use a Jina Flow to deploy your search
application locally or in the cloud using kubernetes.

It is difficult to provide a unified executor that can support all models trained with finetuner,
since finetuner supports different computational frameworks and each model differs in terms
of pre-processing required, input type etc.

There is an alternative though, that can be useful for standardizing your deployment
procedure and unifying different models. Finetuner allows the user to use ONNX for converting
finetuned models to the ONNX format, whether these models have been trained with PyTorch,
TensorFlow or PaddlePaddle. The ONNX format is an open standard for defining
neural network architectures. You can find out more in the [ONNX webpage](https://onnx.ai/).

After converting a trained model to the ONNX format, you can use the
[ONNX encoder](https://hub.jina.ai/executor/2cuinbko) from Jina Hub to deploy your embedding
model. The encoder simply loads your embedding model in ONNX format and uses the ONNX
runtime for running inference. The document tensors are fed to the ONNX model as input
and the output NumPy vectors are assigned to the documents as embeddings.

The ONNX encoder takes care of the embedding part and the only thing left is to add a custom
executor for pre-processing in case there is the need for one. Additionally, you gain
a boost in efficiency, since the conversion to ONNX already optimizes the model in various
ways.

This tutorial will walk you through a simple model finetuning and how you can convert to
ONNX and use the ONNX encoder to deploy a Jina Flow.

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
import finetuner as ft
import torchvision
from docarray import DocumentArray


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
by finetuner. The `onnx` package is required to use this functionality. Also, if
you are using TensorFlow or PaddlePaddle you will need to install additional packages
for the ONNX conversion. Specifically you will need 
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

Now that we have our model exported and have verified that it has the same behaviour as the
original, it's time to deploy it 🚀.


## Deploying using the ONNX Encoder