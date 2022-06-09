# Choose a backbone model

Finetuner provides several widely used backbone models,
including `resnet`, `efficientnet`, `clip` and `bert`.

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
│                                              mlp │            all │          - │          MLP │        Simple MLP encoder trained from scratch │
│                                         resnet50 │ image-to-image │       2048 │          CNN │                         Pretrained on ImageNet │
│                                        resnet152 │ image-to-image │       2048 │          CNN │                         Pretrained on ImageNet │
│                                  efficientnet_b0 │ image-to-image │       1280 │          CNN │                         Pretrained on ImageNet │
│                                  efficientnet_b4 │ image-to-image │       1280 │          CNN │                         Pretrained on ImageNet │
│                     openai/clip-vit-base-patch32 │  text-to-image │        768 │  transformer │       Pretrained on text image pairs by OpenAI │
│                                  bert-base-cased │   text-to-text │        768 │  transformer │ Pretrained on BookCorpus and English Wikipedia │
│ sentence-transformers/msmarco-distilbert-base-v3 │   text-to-text │        768 │  transformer │     Pretrained on Bert, fine-tuned on MS Marco │
└──────────────────────────────────────────────────┴────────────────┴────────────┴──────────────┴────────────────────────────────────────────────┘

```

It should be noted that:

+ resnet/efficientnet models are loaded from [timm](https://github.com/rwightman/pytorch-image-models) library.
+ transformer based models are loaded from huggingface [transformers](https://github.com/huggingface/transformers) library.
+ `msmarco-distilbert-base-v3` has been fine-tuned once by [sentence-transformers](https://www.sbert.net/) on [MS MARCO](https://microsoft.github.io/msmarco/) dataset on top of Bert.