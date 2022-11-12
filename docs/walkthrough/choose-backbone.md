(choose-backbone)=
# Backbone Model

Finetuner provides several widely used backbone models,
including `resnet`, `efficientnet`, `clip` and `bert`.
Thereby, for most of them, Finetuner provides multiple variants, e.g., the common `resnet50 ` and the more complex `resnet152` model.

Finetuner will convert these backbone models to embedding models by removing
the *head* or applying *pooling*,
fine-tuning and producing the final embedding model.
The embedding model can be fine-tuned for text-to-text, image-to-image or text-to-image
search tasks.

You can call:
````{tab} text-to-text
```python
import finetuner

finetuner.describe_models(task='text-to-text')
```
````
````{tab} image-to-image
```python
import finetuner

finetuner.describe_models(task='image-to-image')
```
````
````{tab} text-to-image
```python
import finetuner

finetuner.describe_models(task='text-to-image')
```
````

To get a list of supported models:

````{tab} text-to-text
```bash
import finetuner

finetuner.describe_models(task='text-to-text')
```
````
````{tab} image-to-image
```bash
import finetuner

finetuner.describe_models(task='image-to-image')
```
````
````{tab} text-to-image
```bash
import finetuner

finetuner.describe_models(task='text-to-image')
```
````

+ ResNets are suitable for image-to-image search tasks with high performance requirements, where `resnet152` is bigger and requires higher computational resources than `resnet50`.
+ EfficientNets are suitable for image-to-image search tasks with low training and inference times. The model is more light-weighted than ResNet. Here, `efficientnet_b4` is the bigger and more complex model.
+ CLIP is the one for text-to-image search, where the images do not need to have any text descriptors.
+ BERT is generally suitable for text-to-text search tasks.
+ Msmarco-distilbert-base-v3 is designed for matching web search queries to short text passages and is a suitable backbone for similar text-to-text search tasks.

It should be noted that:

+ resnet/efficientnet models are loaded from the [torchvision](https://pytorch.org/vision/stable/index.html) library.
+ transformer based models are loaded from the huggingface [transformers](https://github.com/huggingface/transformers) library.
+ `msmarco-distilbert-base-v3` has been fine-tuned once by [sentence-transformers](https://www.sbert.net/) on the [MS MARCO](https://microsoft.github.io/msmarco/) dataset on top of BERT.