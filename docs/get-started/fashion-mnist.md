# Finetuning MLP on Image

In this example, we want to "tune" the 32-dim embedding vectors from a 2-layer MLP on the Fashion-MNIST image data, the same dataset that we are using in `jina hello fashion`. 

Precisely, "tuning" means: 
- we set up a Jina search pipeline and will look at the top-K visually similar result;
- we accept or reject the results based on their quality;
- we let the model to remember our feedback and produces better search result.

Hopefully the procedure converges after several rounds; and we get a tuned embedding for better search task.

## Build embedding model

Let's write a 2-layer MLP as our {ref}`embedding model<embedding-model>` using any of the following framework.

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

## Prepare data

Now prepare Fashion-MNIST data for the Finetuner. Note that Finetuner accepts Jina `DocumentArray`/`DocumentArrayMemmap`, so we first convert them into this format.

```python
from finetuner.toydata import generate_fashion_match
```

`fashion_doc_generator` is a generator that yields every image as a `Document` object.

```bash
<jina.types.document.Document id=b9557788-1bab-11ec-a207-1e008a366d49 uri=data:image/png;base64,iVBORw0K... tags={'class': 9.0} blob={'dense': {'buffer': 'AAAAAAAAAAAAAA...==', 'shape': [28, 28], 'dtype': '<f4'}} at 5716974480>
```

## Put together

Finally, let's feed the model and the data into the Finetuner:

```python
import finetuner

finetuner.fit(
   embed_model,
   train_data=generate_fashion_match,
   interactive=True)
```

## Label interactively

You can now label the data by mouse/keyboard. The model will get trained and improved as you are labeling.

```{figure} ../img/labeler-on-fashion-mnist.gif
:align: center
```

From the backend you will see model's training procedure:

```bash
           Flow@22900[I]:🎉 Flow is ready to use!
	🔗 Protocol: 		HTTP
	🏠 Local access:	0.0.0.0:52621
	🔒 Private network:	172.18.1.109:52621
	🌐 Public address:	94.135.231.132:52621
	💬 Swagger UI:		http://localhost:52621/docs
	📚 Redoc:		http://localhost:52621/redoc
           JINA@22900[I]:Finetuner is available at http://localhost:52621/finetuner
UserWarning: The given NumPy array is not writeable, and PyTorch does not support non-writeable tensors. This means you can write to the underlying (supposedly non-writeable) NumPy array using the tensor. You may want to copy the array to protect its data or make it writeable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at  ../torch/csrc/utils/tensor_numpy.cpp:141.) (raised from /Users/hanxiao/Documents/trainer/finetuner/labeler/executor.py:49)
⠴       DONE ━━╸━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 0:00:00 19.0 step/s Loss=2.56 Accuracy=0.33
⠧       DONE ━━╸━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 0:00:00 19.0 step/s Loss=2.65 Accuracy=0.33
⠏       DONE ━━╸━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 0:00:00 19.0 step/s Loss=2.31 Accuracy=0.33
⠙       DONE ━━╸━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 0:00:00 19.7 step/s Loss=2.33 Accuracy=0.33
⠸       DONE ━━╸━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 0:00:00 19.3 step/s Loss=1.18 Accuracy=0.67
```


