# Glossary
```{glossary}

General model
    Any DNN model with no shape restriction on input and output data. For example, classification model, objective detection model, next token prediction model, regression model.

Embedding model
    A DNN with any shape input (image/text/sequence) and an output `ndarray` in the shape `[B x D]`, where `B` is the batch size same as the input, and `D` is the dimension of the embedding.

Unlabeled data
    A `DocumentArray`-like object, filling with `Document` with `.content`.

Labeled data
    A `DocumentArray`-like object, filling with `Document` with `.content` and `.matches`; where each `match` contains `.content` and `.tags['finetuner']['label']`.

Tuner
    A component in Finetuner. Given an {term}`embedding model` and {term}`labeled data`, train the model to fit the data.

Tailor
    A component in Finetuner. Convert any {term}`general model` into an {term}`embedding model`;

Labeler
    A component in Finetuner. Given {term}`unlabeled data` and an {term}`embedding model` or {term}`general model`, labeler asks human for labeling data, trains model and asks better question for labeling.
```