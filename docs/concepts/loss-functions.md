(loss-function)=
# {octicon}`number` Loss Functions

The choice of loss functions relies heavily on your data. In summary, you should utilize loss functions when:

| data type                        | description                                              | loss function                                     | note                                          |
|----------------------------------|----------------------------------------------------------|---------------------------------------------------|-----------------------------------------------|
| article-label                    | Article and it's associated categorical label            | `TripletMarginLoss`, `ArcFaceLoss`, `CosFaceLoss` |                                               |
| text-image pair                  | Text-image pair, text is the descriptor of the image     | `CLIPLoss`                                        |                                               |
| query-article-score              | Query and article with the associated similarity score   | `CosineSimilarityLoss`                            |                                               |
| query-article                    | True pair (matched) query and article                    | `MultipleNegativeRankingLoss`                     |                                               |
| query-article-irrelevant_article | Triplet of query, matched article and irrelevant article | `MarginMSELoss`                                   | use it together with the `synthesis` function |


## TripletMarginLoss

Let's first take a look at our default loss function, `TripletMarginLoss`.  

`TripletMarginLoss` is a *contrastive* loss function, meaning that the loss is calculated by comparing the embeddings of multiple documents (3 to be exact) documents to each other.
Each triplet of documents consists of an anchor document, a positive document and a negative document.
The anchor and the positive document belong to the same class, and the negative document belongs to a different class.
The goal of `TripletMarginLoss` is to maximise the difference between the distance from the anchor to the positive document, and the distance from the anchor to the negative document.
For a more detailed explanation on Triplet Loss, as well as how samples are gathered to create these triplets, see {doc}`/advanced-topics/negative-mining/`.

## ArcFaceLoss and CosFaceLoss

## CLIPLoss

## CosineSimilarityLoss

## MultipleNegativeRankingLoss

## MarginMSELoss

## Negative Mining
