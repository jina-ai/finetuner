# Welcome to Finetuner!

Finetuner allows one to finetune any deep neural network for better embedding on search tasks. It accompanies [Jina](https://github.com/jina-ai/jina) to deliver the last mile of performance-tuning for neural search applications.

Finetuner supports [Pytorch](https://pytorch.org/), [Keras](https://keras.io/) and [PaddlePaddle](https://github.com/PaddlePaddle/Paddle) as the deep learning backend.  

1. Make sure that you have Python 3.7+ installed on Linux/MacOS. You have one of Pytorch, Keras or PaddlePaddle installed.
2. Install Finetuner
   ```bash
   pip install https://github.com/jina-ai/finetuner.git@master
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
   
   finetuner.fit(
       embed_model,
       fashion_doc_generator,  # from finetuner.helloworld.data import fashion_doc_generator
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
:caption: Design
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
:caption: Link
:hidden:

Jina <https://github.com/jina-ai/jina>
Jina Hub <https://hub.jina.ai>
```

---
{ref}`genindex` {ref}`modindex`

