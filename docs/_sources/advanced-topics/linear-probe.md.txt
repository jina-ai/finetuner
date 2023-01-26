(projection-head)=
# {octicon}`pin` Projection Head

## Why freezing?

Depending on your task and the amount of training data,
it is not always necessary to tune the entire model.
In some cases,
freezing some of the weights of the pre-trained model and just fine-tuning specific layers produces comparable or better results.
Furthermore, freezing weights can reduce the training time dramatically.

Finetuner allows you to fine-tune a Linear Projection Head easily.

```{warning}
Currently, we only allow you to freeze layers for image-to-image search tasks.
These models are built on top of Convolutional Neural Networks (CNNs).

For transformer architectures,
we can only fine-tune the entire neural network.
If you need to freeze weights for transformers, consider submitting a feature request in our [Github Issues page](https://github.com/jina-ai/finetuner/issues)
```

```{admonition} Dimensionality reduction
:class: hint
Use a smaller `output_dim` to get compact embeddings.
```

## How?

Finetuner has a built-in module called Tailor.
Given a general model written in Pytorch,
Tailor performs the micro-operations on the model architecture required for fine-tuning and outputs an embedding model.

Given a general model with weights, Tailor performs some or all of the following steps:

+ Iterating over all layers to find dense layers.
+ Chopping off all layers after a certain dense layer.
+ Freezing weights on specific layers.
+ Adding new layers on top of the model.

![tailor](../imgs/tailor.svg)

For example, just using the arguments `freeze=True` and `output_dim=X` with the `fit` function, as shown below:

```diff
run = finetuner.fit(
    model='resnet50',
    ...,
+   freeze=True,
+   output_dim=1024,  # default output_dim of ResNet50 is 2048.
    ...,
)
```

Finetuner will:

1. Remove the classification head of a `ResNet` model, and convert it into an embedding model.
2. Freeze all layers of the embedding model.
3. Attach a trainable 3-layer Linear Projection Head on top of the embedding model with an `output_dim=1024`.

```warning
Keep in mind that whenever you use `freeze=True`, always set `output_dim`.
Otherwise, nothing can be tuned since all layers are frozen.
```

## Summary

If you want to achieve efficient fine-tuning without retraining the entire model,
tuning a Linear Projection Head could be a good solution.