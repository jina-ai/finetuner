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
2. In this example, we want to tune the 32-dim embedding vectors from a 2-layer MLP on the Fashion-MNIST dataset. Let's write a model with any of the following frameworks:
   ````{tab} PyTorch
   
   ```python
   import torch
   
   embed_model = torch.nn.Sequential(
       torch.nn.Flatten(),
       torch.nn.Linear(
           in_features=28 * 28,
           out_features=128,
       ),
       torch.nn.ReLU(),
       torch.nn.Linear(in_features=128, out_features=32))
   ```
   
   ````
   ````{tab} Keras
   ```python
   import tensorflow as tf
   
   embed_model = tf.keras.Sequential([
       tf.keras.layers.Flatten(input_shape=(28, 28)),
       tf.keras.layers.Dense(128, activation='relu'),
       tf.keras.layers.Dense(32)])
   ```
   ````
   ````{tab} Paddle
   ```python
   import paddle
   
   embed_model = paddle.nn.Sequential(
       paddle.nn.Flatten(),
       paddle.nn.Linear(
           in_features=28 * 28,
           out_features=128,
       ),
       paddle.nn.ReLU(),
       paddle.nn.Linear(in_features=128, out_features=32))
   ```
   ````
3. Now feed the model and Fashion-MNIST data into Finetuner.
   ```python
   import finetuner
   from finetuner.toydata import generate_fashion
   
   finetuner.fit(
       embed_model,
       generate_fashion, 
       interactive=True)
   ```

4. You can now label the data in an interactive way. The model will get tuned and improved as you are labeling.
   
   ````{tab} Frontend
   ```{figure} img/labeler-on-fashion-mnist.gif
   :align: center
   ```
   ````
   
   ````{tab} Backend
   ```{figure} img/labeler-backend-on-fashion-mnist.gif
   :align: center
   ```
   ````

Now that you’re set up, let’s dive into more of how Finetuner works and improves the performance of your neural search apps.


## Next steps

<!-- start fit-method -->
Finetuner is extremely easy to learn: all you need is `finetuner.fit()`!

Answer the questions below and quickly find out what you need to learn:

::::{grid} 2
:gutter: 3

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

:::
:::{grid-item-card} Do you have labeled data?

<div>
  <input type="radio" id="labeled_yes" name="labeled" value="0"
         checked>
  <label for="labeled_yes">Yes</label>
</div>

<div>
  <input type="radio" id="labeled_no" name="labeled" value="1">
  <label for="labeled_no">No</label>
</div>

+++
Learn more about {term}`labeled data`.


:::
::::

<div class="usage-card" id="usage-00" style="display: block">

:::{card} Finetuner usage 1

Perfect! Now `embed_model` and `train_data` are already provided by you, simply do:

```python
import finetuner

tuned_model, summary = finetuner.fit(
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

tuned_model, summary = finetuner.fit(
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

tuned_model, summary = finetuner.fit(
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

tuned_model, summary = finetuner.fit(
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
get-started/fashion-mnist
get-started/covid-qa
get-started/celeba
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

