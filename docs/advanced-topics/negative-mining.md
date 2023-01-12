(negative-mining)=
# {octicon}`telescope` Negative Mining

Negative Mining is an advanced technique, which optimizes the way data is sampled from you training dataset.
Usually, it aims at making the metric learning tasks for the model harder during the training. 
In this way it can lead to better fine-tuning results.

## Context: Deep Metric Learning

To understand negative mining, we first take a look at how the training data 
construction is done for metric learning tasks.

Metric Learning algorithms attempt to teach neural network models to tell
which objects are semantically/visually similar and which ones are not.

For uni-modal fine-tuning tasks such as text-to-text, image-to-image, or mesh-to-mesh,
Finetuner constructs training data in the following way:

![batch-sample](../imgs/batch-sampling.png)

Assume we have a list of Documents belonging to four classes: `1`, `2`, `3`, and `4`,
Finetuner will evenly sample *X* items per class to make a batch *B*.

In the next step, arranges the items.
Many loss functions used by Finetuner are calculated based on the embeddings of Triples. 
For those, Finetuner creates all possible Triplets *(anchor, pos, neg)* from this batch.
These Triplets become the training data.
The objective is to pull Documents that belong to the same class together,
while pushing the Documents which belong to a different class away from each other.

![training](../imgs/metric-train.png)


## The Triplet Margin Miner

After Triplets construction,
all possible *(anchor, pos, neg)* Triplets becomes the training data.
However, what if the pre-trained model already performs well on some of the Triplets?

To put it in another way: *what if the distance between `anchor` and `pos` is much smaller than
the distance between `anchor` and `neg`?*

These Triplets, does not contribute to improving the model.
A more effective way is to use hard/semi-hard negative samples for model training.

![mining](../imgs/mining.png)

Let's say `1₀` is an `anchor`, `1₁` is the `pos` while `2₄` is the `neg`, if:

+ `D(anchor, neg) < D(anchor, pos) `, then `neg` can be considered as a "hard negative" (`2₄ - H`).
+ `D(anchor, pos) < D(anchor, neg) < D(anchor, pos) + margin`, where `neg` is a little further from the `pos`, but within the margin, then `neg` can be considered as a "semi-hard negative" (`2₄ - S`).
+ `D(anchor, neg) > D(anchor, pos) + margin`, then `neg` can be considered as "easy negative" (`2₄ - E`).

Usually the training is more effective, when using only hard negatives (and semi-hard negatives) while filtering out easy negatives.

## How?

Finetuner allows you to use miners provided by the [PyTorch Metric Learning](https://kevinmusgrave.github.io/pytorch-metric-learning) framework.
To select a specific miner, you can pass its name to the fit function, e.g., `AngularMiner`, `TripletMarginMiner`, ...

Please note that the miner has to be compatible with the loss function you selected.
For instance, if you choose to train a model with the `TripleMarginLoss`, you can use the `TripletMarginMiner`.
While without this miner, all possible triples with an anchor, a positive, and a negative candidate are constructed, the miner reduces this set of triples.
Usually, only triples with hard negatives are selected where the distance between the positive and the negative example is inside a margin of `0.2`.
If you want to pass additional parameters to configure the miner, you can specify the `miner_options` parameter of the fit function.
The example below shows how to apply hard-negative mining:

```diff
run = finetuner.fit(
    ...,
    loss='TripleMarginLoss',
+   miner='TripletMarginMiner',
+   miner_options={'margin': 0.3, 'type_of_triplets': 'hard'}
)
```

The possible choices `type_of_triplets` are:

+ `easy`: Use all easy triplets - all triplets that do not violate the margin.
+ `semihard`: Use semi-hard triplets, but not hard triples - the negative is violating the margin but not as similar as the positve example.
+ `hard`: Use only hard triplets - the negative is closer to the anchor than the positive.
+ `all`: Use `hard` and `semihard` triples - all but the `easy` triples

Finetuner takes `TripleMarginLoss` as default loss function with no negative mining.
A detailed description of the miners and their parameters is specified in the [PyTorch Metric Learning documentation](https://kevinmusgrave.github.io/pytorch-metric-learning/miners/).

## Summary

Metric Learning and Triplets are extremely useful for fine-tuning models for similarity search.
Easy Triplets have little impact on improving the model.
Consider using semi-hard/hard Triplets for model tuning.