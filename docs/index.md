# Welcome to Finetuner!

```{include} ../README.md
:start-after: <!-- start elevator-pitch -->
:end-before: <!-- end elevator-pitch -->
```

## Quick start

1. Make sure that you have Python 3.7+ installed on Linux/MacOS. You have one of PyTorch (>=1.9), Tensorflow (>=2.5) or PaddlePaddle installed.
   ```bash
   pip install finetuner
   ```
2. In this example, we want to tune the embedding vectors from a ResNet18 on a [customized Celeba dataset](https://static.jina.ai/celeba/celeba-img.zip).Finetuner accepts docarray DocumentArray, so we load CelebA image into this format:
   ```python
   from docarray import DocumentArray

   # please change the file path to your data path
   data = DocumentArray.from_files('img_align_celeba/*.jpg')


   def preproc(doc):
       return (
           doc.load_uri_to_image_tensor(224, 224)
           .set_image_tensor_normalization()
           .set_image_tensor_channel_axis(-1, 0)
       )  # No need for changing channel axes line if you are using tf/keras

   data.apply(preproc)
   ```
3. Let's write a model with any of the following frameworks:
   ````{tab} PyTorch
   
   ```python
   import torchvision
   
   resnet = torchvision.models.resnet18(pretrained=True)
   ```
   
   ````
   ````{tab} Keras
   ```python
   import tensorflow as tf
   
   resnet = tf.keras.applications.resnet18.ResNet18(weights='imagenet')
   ```
   ````
   ````{tab} Paddle
   ```python
   import paddle
   
   resnet = paddle.vision.models.resnet18(pretrained=True)
   ```
   ````
4. Now feed the model and Celeba data into Finetuner.
   ```python
   import finetuner as ft

   tuned_model = ft.fit(
       model=resnet,
       train_data=data,
       loss='TripletLoss',
       epochs=20,
       device='cuda',
       batch_size=128,
       to_embedding_model=True,
       input_size=(3, 224, 224),
       layer_name='adaptiveavgpool2d_67', # layer before fc as feature extractor
       freeze=False,
   )
   ```

Now that you’re set up, let’s dive into more of how Finetuner works and improves the performance of your neural search apps.


## Next steps

<!-- start fit-method -->
Finetuner is extremely easy to learn: all you need is `finetuner.fit()`!



```{include} ../README.md
:start-after: <!-- start support-pitch -->
:end-before: <!-- end support-pitch -->
```

```{toctree}
:caption: Get Started
:hidden:

get-started/swiss-roll/index
get-started/totally-looks-like/index
get-started/clinc150/index
get-started/3d-mesh/index
```


```{toctree}
:caption: Basics
:hidden:

basics/index
```

```{toctree}
:caption: Components
:hidden:

components/index
```

```{toctree}
:caption: Developer Reference
:hidden:
:maxdepth: 1

api/finetuner
```

---
{ref}`genindex` {ref}`modindex`

