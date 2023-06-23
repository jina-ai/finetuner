(loss-function)=
# {octicon}`number` Loss Functions

The choice of loss functions relies heavily on your data. In summary, you should utilize loss functions when:

| data type                        |                                               | loss function                                     | note                                          |
|----------------------------------|--------------------------------------------------------|---------------------------------------------------|-----------------------------------------------|
| article-label                    |          `TripletMarginLoss`, `ArcFaceLoss`, `CosFaceLoss` |                                               |
| text-image pair                  |      `CLIPLoss`                                        |                                               |
| query-article-score              |    `CosineSimilarityLoss`                            |                                               |
| query-article                    |                     `MultipleNegativeRankingLoss`                  |                                               |
| query-article-irrelevant_article |  `MarginMSELoss`                                   | use it together with the `synthesis` function |


## TripletMarginLoss

`TripletMarginLoss` is a *contrastive* loss function, meaning that the loss is calculated by comparing the embeddings of multiple documents (3 to be exact) documents to each other.
Each triplet of documents consists of an anchor document, a positive document and a negative document.
The anchor and the positive document belong to the same class, and the negative document belongs to a different class.
The goal of `TripletMarginLoss` is to maximise the difference between the distance from the anchor to the positive document, and the distance from the anchor to the negative document.

## ArcFaceLoss and CosFaceLoss

SphereFace loss is a loss function that was first formulated for computer vision and face recognition tasks.
Finetuner supports two variations of this loss function, `ArcFaceLoss` and `CosFaceLoss`.
Instead of attempting to minimise the distance between positive pairs and maximise the distance between negative pairs, the SphereFace loss functions compare each sample with an estimate of the center point of each classes' embeddings.
and attempt to minimize the *angular distance* between the document and its class centroid, and maximise the angular distance between the document and the centroids of the other classes.

The `ArcFaceLoss` and `CosFaceLoss` both deviate from the traditional SphereFace loss by including a margin and scaling parameter, which can be used to increase the boundary between each class.
If an item's embedding is within the boundary of the class it belongs to, then no loss is incurred. Choosing appropriate values for the margin and scaling parameter is very important for effective training.
For more information on how `ArcFaceLoss` and `CosFaceLoss` calculate loss, and how these parameters affect the output, see this article on [loss metrics for deep learning](https://hav4ik.github.io/articles/deep-metric-learning-survey#cosface).  

`TripletMarginLoss` uses a `ClassSampler` to construct batches with an equal number of samples of each class in the batch. However, since only one sample is needed to calculate the loss with the `ArcFaceLoss` and `CosFaceLoss` functions, there are no constraints on what each batch needs to contain.
Therefore we can construct batches using random sampling, which is a much simpler and less time consuming method.
By default, runs created using `ArcFaceLoss` or `CosfaceLoss` will use random sampling, however you can specify which type of sampling method you would like to use like so:

```diff
run = finetuner.fit(
    ...,
    loss='ArcFaceLoss',
+   sampler='random'      # use random sampling
+   sampler='class'       # use class sampling
+   sampler='auto'        # infer sampling method based on the loss function (default)

)
```

In cases where the chosen loss function is a form of contrastive loss, such as the default `TripletMarginLoss`, or the `ClipLoss` function (the loss function used for `text-to-image` tasks), a class sampler is needed to properly function.
In these cases, this `sampler` parameters is ignored and the `ClassSampler` is always used.

## CLIPLoss

The `CLIPLoss` is designed to maximize the agreement between image-text pairs that are semantically related,
while minimizing the agreement between pairs that are unrelated.
This encourages the model to learn a joint representation space where images and texts with similar meanings are close together.
The loss function measures how well the CLIP model can encode the similarity between an image and its associated text.
It uses a measure called cosine similarity,
which calculates the cosine of the angle between two vectors.
The higher the cosine similarity value, the closer the vectors are in the joint representation space.

## CosineSimilarityLoss

`CosineSimilarityLoss` is a regression loss function,
which is calculated by comparing the cosine similarity of two embeddings against their ground-truth cosine similarity or some other numerical measure of similarity in the range of 0.0 (completely different) to 1.0 (identical). 
The goal of `CosineSimilarityLoss` is to minimize the MSE (mean squared error) between document pair's cosine score and their ground-truth expected similarity, in order to optimize the model for semantic relatedness, i.e. between images and sentences that describe them in part, between documents that have similar content, etc.

## MultipleNegativeRankingLoss



## MarginMSELoss



## Negative Mining
