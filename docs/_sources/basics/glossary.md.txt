# Glossary
```{glossary}

General model
    Any DNN model with no shape restriction on input and output data. For example, classification model, object detection model, next token prediction model, regression model.

Embedding model
    A DNN with any shape input (image/text/sequence) and an output `ndarray` in the shape `[B x D]`, where `B` is the batch size same as the input, and `D` is the dimension of the embedding.

Unlabeled dataset
    A `DocumentArray`-like object, filled with `Document`s with `.content`.

Labeled dataset
    A `DocumentArray`-like object, filled with `Document`s with `.content`, where documents also contain some kind of labels that can be used for training an {term}`embedding model`.
    
Class dataset
    A kind of {term}`labeled dataset`, where each `Document` has a class label stored in `.tags['finetuner_label']`, and does not have `.matches`.

Session dataset
    A kind of {term}`labeled dataset`, where each root `Document` contains `.matches` and no label; its matches contain label saved under `.tags['finetuner_label']`. That label can be either 1 (for a match similar to its reference `Document`) or -1 (for match dissimilar to its reference `Document`).

Tuner
    A component in Finetuner. Given an {term}`embedding model` and a {term}`labeled dataset`, it trains the model to fit the data.

Tailor
    A component in Finetuner. Converts any {term}`general model` into an {term}`embedding model`;

Labeler
    A component in Finetuner. Given an {term}`unlabeled dataset` and an {term}`embedding model` or {term}`general model`, the labeler asks the annotator for labels, while training the model in the background and constantly optimizing the questions presented to the annotator.
```
