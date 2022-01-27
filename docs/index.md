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

Answer the questions below and quickly find out what you need to learn:

:::{grid-item-card} Do you have an embedding model?

<div>
  <input type="radio" id="embed_model_yes" name="embed_model" value="0"
         checked>
  <label for="embed_model_yes">Yes</label>
</div>

<div>
  <input type="radio" id="embed_model_no" name="embed_model" value="1">
  <label for="embed_model_no">No</label>
</div>

+++
Learn more about {term}`embedding model`.


<div class="usage-card" id="usage-00" style="display: block">

:::{card} Finetuner usage 1

Perfect! Now that `embed_model` and `train_data` are already provided by you, simply do:

```python
import finetuner

tuned_model = finetuner.fit(
    embed_model,
    train_data=train_data
)
```

+++
Learn more about {term}`Tuner`.
:::

</div>
<div class="usage-card" id="usage-01">

:::{card} Finetuner usage 2

You have an `embed_model` to use, but no labeled data for fine-tuning this model. No worries, you can use Finetuner to interactively label data and train `embed_model` as follows:

```{code-block} python
---
emphasize-lines: 6
---
import finetuner

tuned_model = finetuner.fit(
    embed_model,
    train_data=unlabeled_data,
    interactive=True
)
```

+++
Learn more about {term}`Tuner` and {term}`Labeler`.
:::

</div>
<div class="usage-card" id="usage-10">

:::{card} Finetuner usage 3

You have a `general_model` but it does not output embeddings. Luckily, you've got some `labeled_data` for training. No worries, Finetuner can convert your model into an embedding model and train it via: 

```{code-block} python
---
emphasize-lines: 6, 7
---
import finetuner

tuned_model = finetuner.fit(
    general_model,
    train_data=labeled_data,
    to_embedding_model=True,
    freeze=False,
)
```

+++
Learn more about {term}`Tailor` and {term}`Tuner`.
:::

</div>
<div class="usage-card" id="usage-11">

:::{card} Finetuner usage 4

You have a `general_model` which is not for embeddings. Meanwhile, you don't have any labeled data for training. But no worries, Finetuner can help you train an embedding model with interactive labeling on-the-fly: 

```{code-block} python
---
emphasize-lines: 6, 7
---
import finetuner

tuned_model = finetuner.fit(
    general_model,
    train_data=labeled_data,
    interactive=True,
    to_embedding_model=True,
    freeze=False,
)
```

+++
Learn more about {term}`Tailor`, {term}`Tuner` and {term}`Labeler`.
:::

</div>

<script>
    function init() {
        document.getElementById('embed_model_yes').click();
        document.getElementById('labeled_yes').click()
    }
    window.onload = init;
    function myfunction(event) {
        const answer = document.querySelector('input[name="embed_model"]:checked').value +document.querySelector('input[name="labeled"]:checked').value;
         document.querySelectorAll(".usage-card").forEach((input) => {
                 input.style.display= 'None'
             });
        document.getElementById("usage-"+answer).style.display = 'block'
    }
    document.querySelectorAll("input[name='embed_model']").forEach((input) => {
        input.addEventListener('change', myfunction);
    });
    document.querySelectorAll("input[name='labeled']").forEach((input) => {
        input.addEventListener('change', myfunction);
    });
</script>

<!-- end fit-method -->

```{include} ../README.md
:start-after: <!-- start support-pitch -->
:end-before: <!-- end support-pitch -->
```

```{toctree}
:caption: Get Started
:hidden:

get-started/swiss-roll/index
get-started/totally-looks-like/index
get-started/covid-qa
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

