# Loss and Miners

When training your embedding model, it is crucial that the model learns to embed similar items close together in the latent space, and different items far from each other.

For that, you can train your model using the [Triplet or Siamese loss](#triplet-and-siamese-loss). By default, these losses will look at all possible triplets/pairs of embeddings in the batch. As we can expect there to be many "easy" triplets/pairs, you may want to select only the hard ones, to make the model learn faster. For that, you can use [tuple mining](#tuple-mining).

## Triplet and Siamese Loss



## Tuple Miners

