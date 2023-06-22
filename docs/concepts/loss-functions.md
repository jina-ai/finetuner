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

## ArcFaceLoss and CosFaceLoss

## CLIPLoss

## CosineSimilarityLoss

## MultipleNegativeRankingLoss

## MarginMSELoss

## Negative Mining
