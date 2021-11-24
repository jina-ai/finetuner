# Glossary
```{glossary}

General model
    Any DNN model with no shape restriction on input and output data. For example, classification model, objective detection model, next token prediction model, regression model.

Embedding model
    A DNN with any shape input (image/text/sequence) and an output `ndarray` in the shape `[B x D]`, where `B` is the batch size same as the input, and `D` is the dimension of the embedding.

Unlabeled dataset
    A `DocumentArray`-like object, filled with `Document`s with `.content`.

Labeled dataset
    A `DocumentArray`-like object, filled with `Document`s with `.content`, where documents also contain some kind of labeles that can be used for training an {term}`embedding model`.
    
Class dataset
    A kind of {term}`labeled dataset`, where each `Document` has a class label stored in `.tags['finetuner']['label']`, and does not have `.matches`

Session dataset
    A kind of {term}`labeled dataset`, where each root `Document` contains `.matches` and no label; its matches contain label saved under `.tags['finetuner']['label']`. That label can be either 1 (for a match similar to its reference `Document`) or -1 (for match dissimilar from its reference `Document`)

Tuner
    A component in Finetuner. Given an {term}`embedding model` and {term}`labeled dataset`, it trains the model to fit the data.

Tailor
    A component in Finetuner. Converts any {term}`general model` into an {term}`embedding model`;

Labeler
    A component in Finetuner. Given {term}`unlabeled dataset` and an {term}`embedding model` or {term}`general model`, labeler asks human for labeling data, trains model and asks better question for labeling.
```
