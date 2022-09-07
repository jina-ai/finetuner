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
```python
import finetuner

finetuner.describe_models()
```

To get a list of supported models:

```bash
                                                               Finetuner backbones                                                                
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                            model ┃           task ┃ output_dim ┃ architecture ┃                                    description ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│                                         resnet50 │ image-to-image │       2048 │          CNN │                         Pretrained on ImageNet │
│                                        resnet152 │ image-to-image │       2048 │          CNN │                         Pretrained on ImageNet │
│                                  efficientnet_b0 │ image-to-image │       1280 │          CNN │                         Pretrained on ImageNet │
│                                  efficientnet_b4 │ image-to-image │       1280 │          CNN │                         Pretrained on ImageNet │
│                     openai/clip-vit-base-patch32 │  text-to-image │        768 │  transformer │       Pretrained on text image pairs by OpenAI │
│                                  bert-base-cased │   text-to-text │        768 │  transformer │ Pretrained on BookCorpus and English Wikipedia │
│ sentence-transformers/msmarco-distilbert-base-v3 │   text-to-text │        768 │  transformer │     Pretrained on BERT, fine-tuned on MS Marco │
└──────────────────────────────────────────────────┴────────────────┴────────────┴──────────────┴────────────────────────────────────────────────┘

```
+ ResNets are suitable for image-to-image search tasks with high performance requirements, where `resnet152` is bigger and requires higher computational resources than `resnet50`.
+ EfficientNets are suitable for image-to-image search tasks with low training and inference times. The model is more light-weighted than ResNet. Here, `efficientnet_b4` is the bigger and more complex model.
+ CLIP is the one for text-to-image search, where the images do not need to have any text descriptors.
+ BERT is generally suitable for text-to-text search tasks.
+ Msmarco-distilbert-base-v3 is designed for matching web search queries to short text passages and is a suitable backbone for similar text-to-text search tasks.

It should be noted that:

+ resnet/efficientnet models are loaded from the [torchvision](https://pytorch.org/vision/stable/index.html) library.
+ transformer based models are loaded from the huggingface [transformers](https://github.com/huggingface/transformers) library.
+ `msmarco-distilbert-base-v3` has been fine-tuned once by [sentence-transformers](https://www.sbert.net/) on the [MS MARCO](https://microsoft.github.io/msmarco/) dataset on top of BERT.