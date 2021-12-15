# Loss and Miners

When training your embedding model, it is crucial that the model learns to embed similar items close together in the latent space, and different items far from each other.

For that, you can train your model using the [Triplet or Siamese loss](#triplet-and-siamese-loss). By default, these losses will look at all possible triplets/pairs of embeddings in the batch. As we can expect there to be many "easy" triplets/pairs, you may want to select only the hard ones, to make the model learn faster. For that, you can use [tuple mining](#tuple-mining).

## Triplet and Siamese Loss

Triplet and Siamese loss both compare the distances between the embeddings of similar and dissimilar items, and create a loss that penalizes similar items being too far apart, or dissimilar ones being too close to each other.

### Triplet Loss

Triplet loss works with a triplet composed of an anchor, a positive sample (item similar to the anchor) and a negative sample (item dis-similar from the anchor). The loss for a single such triplet is then computed as

$$\ell_{i, p, n}=\max(0, d(\mathbf{x}_i, \mathbf{x}_p)-d(\mathbf{x}_i, \mathbf{x}_n)+m)$$

Here $\mathbf{x}_i, \mathbf{x}_p, \mathbf{x}_n$ and the embeddings of the anchor, positive and negative sample, respectively, $d$ is a distance function, and $m$ is a desired wedge between the distances of similar and dis-similar items.

### Siamese Loss

Siamese loss works with a tuple of items, an anchor and either a positive or a negative sample. The loss for a single pair is computed as

$$\ell_{i,j} = \mathrm{sim}(i,j)d(\mathbf{x}_i, \mathbf{x}_j) + (1 - \mathrm{sim}(i,j))\max(m - d(\mathbf{x}_i, \mathbf{x}_j))$$

Here $\mathrm{sim}(i,j)$ denotes the similarity function, which returns 1 if items $i$ and $j$ are similar, and 0 if they are dis-similar.

### Use with Tuner

It's straightforward to use these loss functions with the Tuner. You can just give the name of the loss function (as a string) as the `loss` argument on initialization, or you can instantiate the loss object, which allows you to customize parameters (including [miners](#tuple-miners))

````{tab} Pytorch
```python
from finetuner.tuner.pytorch import PytorchTuner
from finetuner.tuner.pytorch.losses import TripletLoss

loss = TripletLoss(distance='cosine', margin=0.5)

tuner = PytorchTuner(..., loss=loss)
```
````
````{tab} Keras
```python
from finetuner.tuner.keras import KerasTuner
from finetuner.tuner.keras.losses import TripletLoss

loss = TripletLoss(distance='cosine', margin=0.5)

tuner = KerasTuner(..., loss=loss)
```
````
````{tab} Paddle
```python
from finetuner.tuner.paddle import PaddleTuner
from finetuner.tuner.paddle.losses import TripletLoss

loss = TripletLoss(distance='cosine', margin=0.5)

tuner = PaddleTuner(..., loss=loss)
```
````

## Tuple Miners

