(loss functions)=
# {octicon}`pin` Loss Functions

Loss functions are used to calculate the quality of embeddings while training; the higher the output of the loss function, the more the parameters of the model will be updated.
By default we use Triplet Margin Loss, however we also support two more advanced loss functions, ArcFace Loss and CosFace Loss.

## Triplet Margin Loss

Lets first take a look at our default loss function, Triplet Margin Loss.  

Triplet Margin Loss is a *contrastive* loss function, meaning that the loss is calculated by comparing the embeddings of multiple documents, three, to be exact.
Each triplet of documents consists of an anchor document, a positive document and a negative document.
The anchor and the positive document belong to the same class, and the negative document belongs to a different class.
The goal of Triplet margin loss is to maximise the difference between the distance from the anchor to the positive document, and the distance from the anchor to the negative document.
For a more detailed explanation on how the loss is calculated, as well as how samples are gathered to create these Triplets, see {doc}`/advanced_topics/negative-mining`.

## SphereFace losses

The other two loss functions that we support, ArcFace and CosFace, are both variations on the SphereFace loss function. Instead of attempting to minimise the distance between positive pairs and maximise the distance between negative pairs, the SphereFace loss functions compare each sample with the centroid of each class and attempts to minimize the *angular distance* between the document and its class centroid, and maximise the angular distance between the document and the centroids of the other classes.

![training](../imgs/SphereFace-training.png)

The ArcFace and CosFace both adjust 