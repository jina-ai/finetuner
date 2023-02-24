(advanced-losses-optimizers-poolers)=
# {octicon}`mortar-board` Advanced losses and optimizers
Many of the models supported by Finetuner use of similar methods during finetuning, like methods for *calculating loss*, *sampling* and *pooling* . Finetuner offers alternative methods for each of these tasks, and in some cases, choosing specific methods can improve Finetuner performance.

## Loss functions

Loss functions are used to calculate the quality of embeddings while training; the higher the output of the loss function, the larger the update to the parameters of the model.
By default we use `TripletMarginLoss`, however we support many other loss functions as well, including `ArcFaceLoss` and `CosFaceLoss`.

```{Important}
Please check the [developer reference](../../api/finetuner/#finetuner.fit) to get the available options for `loss`.
```

### TripletMarginLoss

Let's first take a look at our default loss function, `TripletMarginLoss`.  

`TripletMarginLoss` is a *contrastive* loss function, meaning that the loss is calculated by comparing the embeddings of multiple documents (3 to be exact) documents to each other.
Each triplet of documents consists of an anchor document, a positive document and a negative document.
The anchor and the positive document belong to the same class, and the negative document belongs to a different class.
The goal of `TripletMarginLoss` is to maximise the difference between the distance from the anchor to the positive document, and the distance from the anchor to the negative document.
For a more detailed explanation on Triplet Loss, as well as how samples are gathered to create these triplets, see {doc}`/advanced-topics/negative-mining/`.

### SphereFace losses

SphereFace loss is a loss function that was first formulated for computer vision and face recognition tasks.
Finetuner supports two variations of this loss function, `ArcFaceLoss` and `CosFaceLoss`.
Instead of attempting to minimise the distance between positive pairs and maximise the distance between negative pairs, the SphereFace loss functions compare each sample with an estimate of the center point of each classes' embeddings.
and attempt to minimize the *angular distance* between the document and its class centroid, and maximise the angular distance between the document and the centroids of the other classes.

![training](../imgs/SphereFace-training.png)

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

### Using an optimizer

In order to keep track and refine our estimation of the class centers across batches, these SphereFace loss functions require an additional optimizer during training.
By default, the type of optimizer used will be the same as the one used for the model itself, but you can also choose a different optimizer for your loss function using the `loss_optimizer` parameter.
The list of available optimizers are discussed in the [run job](../walkthrough/run-job.md) section.

```diff
run = finetuner.fit(
    ...,
    loss='ArcFaceLoss',
+   loss_optimizer='Adam',
+   loss_optimizer_options={'weight_decay': 0.01}
)
```

### Comparing with TripletMarginLoss

Using these loss functions over the default `TripletMarginLoss` can result in clearer divisions between the domains representing each class in the embedding space.
As an example, the figure below shows the domains of the 10 classes of the [FMNIST dataset](https://github.com/zalandoresearch/fashion-mnist) projected onto 2D space using the `umap` library after training with `TripletMarginLoss`, `ArcFaceLoss` and `CosFaceLoss`.

![distributions-loss](../imgs/distributions-loss.png)

Each color represents a different class. You can see how all of the loss functions are able to separate some of the classes from the others,
but struggle to separate the green, blue, pink, purple and red classes,
with `TripletMarginLoss` sperarating them the least, and `ArcFaceLoss` separating them the most.

### Choosing an optimizer

While `ArcFaceLoss` and `CosFaceLoss` are capable of outperforming `TripletMarginLoss`, There are some cases where the opposite is the case.  
In cases where each class contains very few examples of each class, but many classes, `ArcFaceLoss` and `CosFaceLoss` struggle to separate every class,
and doesn't perform as well as `TripletMarginLss`.  
Another case where `ArcFaceLoss` and `CosFaceLoss` struggle is when attempting to embed documents of a class they were not trained on. 
Since these loss functions operate by creating centroids, and then attempting to 'classify' input documents by embedding them in a space close to a centroid, 
data that does not belong to a class with an existing centroid may erroneously be given an embedding similar to that of a class it does not belong to.  
In cases such as these, `TripletMarginLoss` may result in a greater improvement in performance thatn `ArcFaceLoss` or `CosFaceLoss`.