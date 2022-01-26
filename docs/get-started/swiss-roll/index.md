# Finetuning MLP on Swiss Roll Dataset

```{tip}
The full Colab notebook can be [found here](https://colab.research.google.com/drive/1zQjA2LAOiS6yWqh_vszJfF1yAZfzxaJZ?usp=sharing), which is self-contained and can be run out of the box.
```

In this example, we use Finetuner to tune the embedding manifold on Swiss roll dataset.

```{figure} swissroll.gif
:width: 50%
```

Swiss roll dataset is a set of 3D points, and has been widely used in classic machine learning literatures on dimensionality reduction and manifold learning. Here we will use it to validate the effectiveness of Finetuner.

## Load data

`scikit-learn` provides a very simple API to generate a swiss roll dataset. One can simply do:

```python
from sklearn.datasets import make_swiss_roll
n_samples = 1500
noise = 0.05
X, y = make_swiss_roll(n_samples, noise=noise)
```

To see how it looks like, we can use `matplotlib`:

````{dropdown} Code for plotting

```python
def minmax_norm(y):
    return (y-y.min())/(y.max()-y.min())

import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.pyplot as plt
fig = plt.figure()
ax = p3.Axes3D(fig)
ax.view_init(7, -80)

ax.scatter(
        X[:,0],
        X[:,1],
        X[:,2],
        color=plt.cm.cool(minmax_norm(y)),
        s=10)
```

````

```{figure} output_4_2.png
:width: 70% 
```


## Nearest neighbour in original space

If we consider each 3D point as a {class}`Document`, then the dataset is basically a {class}`DocumentArray`. We can easily fill the generated data into a `DocumentArray`.

```python
from docarray import DocumentArray
import numpy as np

da = DocumentArray.empty(len(X))
da.tensors = X.astype(np.float32)
```

Now let's consider our 3D coordinates as the embedding space. And then we randomly take a point as the query and search for its 300 nearest neighbours based on the Cosine distance:

```python
da.embeddings = da.tensors

q = da[7]
q.match(da, limit=300, normalize=(1, 0))
```

Plot it and we get the following:

```{figure} output_9_1.png
:width: 70%
```

This result should be very straightforward and intuitive. The red point represents the query and the colored points represent the nearest neighbours. The temperature of the color represents the distance: **the colder the closer, the warmer the further**.

## Warp the space by following the roll

To validate if Finetuner works or not, we need to design an experiment to show: **when keep telling Finetuner that some points need to be "closer", if Finetuner can eventually come up with a model that gives the desired embedding.** This is the logic that guides our experiment design. 

First, we need to design a supervision, which says some points should be together.

Let's use the depth-axis `X[:, 1]` and cut it into 5 parts. Points that fall into each part are labeled accordingly.

```python
bins = np.linspace(0, 1, 5)
label_y = np.digitize(minmax_norm(X[:, 1]), bins)
```

Let's look at how does this partition our dataset. 

```{figure} output_13_2.png
:width: 70%
```

One can see that the full dataset is partitioned **along the depth-axis**. Each partition covers complete roll patterns.

We will use it as the surpervision to finetune a DNN model.

As a result, we expect that when given a query point and ask the finetuned model its nearest neighbours, the model should return all points **in a complete roll**, rather than just intuitively close points as before.

## Call Finetuner

Let's set the label to `.tags` via:

```python
for d, l_y in zip(da, label_y):
    d.tags['finetuner_label'] = int(l_y)
```

Build a simple 3-layer MLP with 6 dimensional embedding space:

```python
import torch

D = 6
embed_model = torch.nn.Sequential(
            torch.nn.Linear(in_features=3, out_features=D),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=D, out_features=D),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=D, out_features=D))
```

And finally call Finetuner:

```python
import finetuner
new_model = finetuner.fit(embed_model, train_data=da, epochs=500)
```

In this simple example, training on GPU should take less than 1 minute. Let's have a look at the
training loss:

```{figure} output_19_0.png
:width: 50%
```

To embed the points into the finetuned embedding space, one can now do

```python
finetuner.embed(da,  new_model)
```

`da.embeddings` are now filled with new embeddings.

## Validate results

To validate the effectiveness of Finetuner, we repeat the nearest neighbours search and plot the results on four different spaces:

1. The original space
2. The embedding space from a random initialized MLP
3. The finetuned embedding space
4. The groundtruth

```python
from copy import deepcopy

da2 = deepcopy(da)
da3 = deepcopy(da)
da4 = deepcopy(da)

finetuner.embed(da2,  embed_model)  #: random init MLP
finetuner.embed(da3, new_model)  #: finetuned space
da4.embeddings = np.expand_dims(label_y, axis=1).astype(np.float32)  #: groundtruth
```

Let's randomly sample four take a look on the results:

```{figure} output_25_0.png
:width: 80%
```
```{figure} output_25_1.png
:width: 80%
```
```{figure} output_25_2.png
:width: 80%
```
```{figure} output_25_3.png
:width: 80%
```

Pretty good, right? Note that the conclusion of "good" comes from the observation that the third column gives the nearest neighbours along the full roll (i.e. more similar to the fourth column); and it does not retrieve trivial nearest neighbours as in the first column, which suggests that Finetuner is effective.


