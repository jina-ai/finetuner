# Welcome to Finetuner!

Finetuner allows one to finetune any deep Neural Network for better embedding on search tasks. It is designed to accompany [Jina](https://github.com/jina-ai/jina) on delivering the last mile of performance-tuning for neural search applications.

Finetuner supports [Pytorch](https://pytorch.org/), [Keras](https://keras.io/) and [PaddlePaddle](https://github.com/PaddlePaddle/Paddle) as the deep learning backend.  

1. Make sure that you have Python 3.7+ installed on Linux/MacOS. You have one of Pytorch, Keras or PaddlePaddle installed.
2. Install Finetuner
   ````{python}
   pip install https://github.com/jina-ai/finetuner.git@master
   ````
4. In this example, we want to tune the 32-dim embedding vectors from a 2-layer MLP on the Fashion-MNIST data. Let's write our model:
   ````{tab} Pytorch
   
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
5. Now feed the model and Fashion-MNIST data into the finetuner.
   ```python
   import finetuner
   
   finetuner.fit(
       embed_model,
       head_layer='CosineLayer',
       unlabeled_data=fashion_doc_generator,  # from finetuner.helloworld.data import fashion_doc_generator
       interactive=True)
   ```

Now that you’re set up, let’s dive into more of how Finetuner works and can help you improve the neural search pipelin performance.


```{toctree}
:caption: Developer Reference
:hidden:
:maxdepth: 1

api/finetuner
```


```{toctree}
:caption: Support
:hidden:

Issue tracker <https://github.com/jina-ai/finetuner/issues>
Slack community <https://slack.jina.ai>
Youtube <http://youtube.com/c/jina-ai>
Twitter @JinaAI_ <https://twitter.com/JinaAI_>
LinkedIn <https://www.linkedin.com/company/jinaai/>
Jina AI Company <https://jina.ai>

```

---
{ref}`genindex` {ref}`modindex`

