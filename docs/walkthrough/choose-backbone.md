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
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                             name ┃           task ┃ output_dim ┃ architecture ┃                                                description ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│                                  bert-base-cased │   text-to-text │        768 │  transformer │ BERT model pre-trained on BookCorpus and English Wikipedia │
│                     openai/clip-vit-base-patch16 │  text-to-image │        512 │  transformer │                         CLIP base model with patch size 16 │
│                     openai/clip-vit-base-patch32 │  text-to-image │        512 │  transformer │                                            CLIP base model │
│                openai/clip-vit-large-patch14-336 │  text-to-image │        768 │  transformer │                        CLIP large model for 336x336 images │
│                    openai/clip-vit-large-patch14 │  text-to-image │       1024 │  transformer │                        CLIP large model with patch size 14 │
│                                  efficientnet_b0 │ image-to-image │       1280 │          cnn │                    EfficientNet B0 pre-trained on ImageNet │
│                                  efficientnet_b4 │ image-to-image │       1792 │          cnn │                    EfficientNet B4 pre-trained on ImageNet │
│                                    RN101::openai │  text-to-image │        512 │  transformer │                            Open CLIP "RN101::openai" model │
│                          RN101-quickgelu::openai │  text-to-image │        512 │  transformer │                  Open CLIP "RN101-quickgelu::openai" model │
│                         RN101-quickgelu::yfcc15m │  text-to-image │        512 │  transformer │                 Open CLIP "RN101-quickgelu::yfcc15m" model │
│                                   RN101::yfcc15m │  text-to-image │        512 │  transformer │                           Open CLIP "RN101::yfcc15m" model │
│                                      RN50::cc12m │  text-to-image │       1024 │  transformer │                              Open CLIP "RN50::cc12m" model │
│                                     RN50::openai │  text-to-image │       1024 │  transformer │                             Open CLIP "RN50::openai" model │
│                            RN50-quickgelu::cc12m │  text-to-image │       1024 │  transformer │                    Open CLIP "RN50-quickgelu::cc12m" model │
│                           RN50-quickgelu::openai │  text-to-image │       1024 │  transformer │                   Open CLIP "RN50-quickgelu::openai" model │
│                          RN50-quickgelu::yfcc15m │  text-to-image │       1024 │  transformer │                  Open CLIP "RN50-quickgelu::yfcc15m" model │
│                                  RN50x16::openai │  text-to-image │        768 │  transformer │                          Open CLIP "RN50x16::openai" model │
│                                   RN50x4::openai │  text-to-image │        640 │  transformer │                           Open CLIP "RN50x4::openai" model │
│                                  RN50x64::openai │  text-to-image │       1024 │  transformer │                          Open CLIP "RN50x64::openai" model │
│                                    RN50::yfcc15m │  text-to-image │       1024 │  transformer │                            Open CLIP "RN50::yfcc15m" model │
│                          ViT-B-16::laion400m_e31 │  text-to-image │        512 │  transformer │                  Open CLIP "ViT-B-16::laion400m_e31" model │
│                          ViT-B-16::laion400m_e32 │  text-to-image │        512 │  transformer │                  Open CLIP "ViT-B-16::laion400m_e32" model │
│                                 ViT-B-16::openai │  text-to-image │        512 │  transformer │                         Open CLIP "ViT-B-16::openai" model │
│                 ViT-B-16-plus-240::laion400m_e31 │  text-to-image │        640 │  transformer │         Open CLIP "ViT-B-16-plus-240::laion400m_e31" model │
│                 ViT-B-16-plus-240::laion400m_e32 │  text-to-image │        640 │  transformer │         Open CLIP "ViT-B-16-plus-240::laion400m_e32" model │
│                            ViT-B-32::laion2b_e16 │  text-to-image │        512 │  transformer │                    Open CLIP "ViT-B-32::laion2b_e16" model │
│                          ViT-B-32::laion400m_e31 │  text-to-image │        512 │  transformer │                  Open CLIP "ViT-B-32::laion400m_e31" model │
│                          ViT-B-32::laion400m_e32 │  text-to-image │        512 │  transformer │                  Open CLIP "ViT-B-32::laion400m_e32" model │
│                                 ViT-B-32::openai │  text-to-image │        512 │  transformer │                         Open CLIP "ViT-B-32::openai" model │
│                ViT-B-32-quickgelu::laion400m_e31 │  text-to-image │        512 │  transformer │        Open CLIP "ViT-B-32-quickgelu::laion400m_e31" model │
│                ViT-B-32-quickgelu::laion400m_e32 │  text-to-image │        512 │  transformer │        Open CLIP "ViT-B-32-quickgelu::laion400m_e32" model │
│                       ViT-B-32-quickgelu::openai │  text-to-image │        512 │  transformer │               Open CLIP "ViT-B-32-quickgelu::openai" model │
│                             ViT-L-14-336::openai │  text-to-image │        768 │  transformer │                     Open CLIP "ViT-L-14-336::openai" model │
│                                 ViT-L-14::openai │  text-to-image │        768 │  transformer │                         Open CLIP "ViT-L-14::openai" model │
│                                        resnet152 │ image-to-image │       2048 │          cnn │                          ResNet152 pre-trained on ImageNet │
│                                         resnet50 │ image-to-image │       2048 │          cnn │                           ResNet50 pre-trained on ImageNet │
│ sentence-transformers/msmarco-distilbert-base-v3 │   text-to-text │        768 │  transformer │                    Pretrained BERT, fine-tuned on MS Marco │
└──────────────────────────────────────────────────┴────────────────┴────────────┴──────────────┴───────────────────────────────────────────────────────────

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