# Welcome to Finetuner!

```{include} ../README.md
:start-after: <!-- start elevator-pitch -->
:end-before: <!-- end elevator-pitch -->
```

## Quick start

1. Make sure that you have Python 3.7+ installed on Linux/MacOS. You have one of Pytorch, Keras or PaddlePaddle installed.
2. Install Finetuner
   ```bash
   pip install finetuner
   ```
3. In this example, we want to tune the 32-dim embedding vectors from a 2-layer MLP on the Fashion-MNIST data. Let's write a model with any of the following framework:
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
4. Now feed the model and Fashion-MNIST data into Finetuner.
   ```python
   import finetuner
   from finetuner.toydata import generate_fashion_match
   
   finetuner.fit(
       embed_model,
       generate_fashion_match, 
       interactive=True)
   ```

5. You can now label the data in an interactive way. The model will get tuned and improved as you are labeling.
   
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

Now that you’re set up, let’s dive into more of how Finetuner works and can improve the performance of your neural search apps.


## Next step

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
Know more about {term}`embedding model`.

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
Know more about {term}`labeled data`.


:::
::::

<div class="usage-card" id="usage-00" style="display: block">

:::{card} Finetuner usage

Perfect! Now `embed_model` and `train_data` are given by you already, simply do:

```python
import finetuner

finetuner.fit(
    embed_model,
    train_data=train_data
)
```

+++
Know more about {term}`Tuner`.
:::

</div>
<div class="usage-card" id="usage-01">

:::{card} Finetuner usage

Perfect! Now `embed_model` and `unlabeled_data` are given by you, you can use Finetuner to interactive label data and train `embed_model` as below:

```python
import finetuner

finetuner.fit(
    embed_model,
    train_data=unlabeled_data,
    interactive=True
)
```

+++
Know more about {term}`Tuner` and {term}`Labeler`.
:::

</div>
<div class="usage-card" id="usage-10">

:::{card} Finetuner usage

You have a `general_model` but it is not yet for embedding. Luckily you provide some `labeled_data` for training, so simply do: 

```python
import finetuner

finetuner.fit(
    general_model,
    train_data=labeled_data,
    to_embedding_model=True,
    output_dim=100
)
```

+++
Know more about {term}`Tailor` and {term}`Tuner`.
:::

</div>
<div class="usage-card" id="usage-11">

:::{card} Finetuner usage

You have a `general_model` but it is not yet for embedding. You only have unlabeled data which is not ready for training. But no worries, Finetuner can help you train an embedding model with interactive labeling on-the-fly: 

```python
import finetuner

finetuner.fit(
    general_model,
    train_data=labeled_data,
    interactive=True,
    to_embedding_model=True,
    output_dim=100
)
```

+++
Know more about {term}`Tailor`, {term}`Tuner` and {term}`Labeler`.
:::

</div>

<script>
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

```{toctree}
:caption: Get Started
:hidden:

get-started/fashion-mnist
get-started/covid-qa
```


```{toctree}
:caption: Design (INTERNAL ONLY)
:hidden:

design/index
```

```{toctree}
:caption: Basics
:hidden:

basics/index
```


```{toctree}
:caption: Developer Reference
:hidden:
:maxdepth: 1

api/finetuner
```


```{toctree}
:caption: Ecosystem
:hidden:

Jina <https://github.com/jina-ai/jina>
Jina Hub <https://hub.jina.ai>
```

---
{ref}`genindex` {ref}`modindex`

