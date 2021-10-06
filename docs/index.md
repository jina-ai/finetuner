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
4. Now feed the model and Fashion-MNIST data into the finetuner.
   ```python
   import finetuner
   from finetuner.toydata import generate_fashion_match
   
   finetuner.fit(
       embed_model,
       generate_fashion_match(num_pos=0, num_neg=0),  #: no synthetic positive & negative examples 
       interactive=True)
   ```

5. You can now label the data in an interactive way. The model will get finetuned and improved as you are labeling.
   
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

