(choose-backbone)=
# Backbone Model

Finetuner provides several widely used backbone models,
including `resnet`, `efficientnet`, `clip`, `bert` and `pointnet`.
Each of these models have their own variants and serve a specific task.

Finetuner will convert these backbone models to embedding models by removing
the *head* or applying *pooling*,
performing fine-tuning and producing the final embedding model.
The embedding model can be fine-tuned for text-to-text, image-to-image, text-to-image or mesh-to-mesh
search tasks.

## Text-to-Text Models

We support two different variations of the `bert` model for text-to-text encoding tasks.
- `bert-base-en` is a more general purpose model, which will be suitable for most text-to-text search tasks.
- `sbert-base-en` has been fine-tuned once by [sentence-transformers](https://www.sbert.net/) on the 
  [MS MARCO](https://microsoft.github.io/msmarco/) dataset on top of BERT.
  It is designed for matching web search queries to short text passages and 
  is a suitable backbone for similar text-to-text search tasks.

These two models have the same `output_dim` meaning that the shape of the embeddings produced are the same.
This means that, while the performance of these models may differ depending on the domain, they can be used interchangably.

```bash
                                           Finetuner backbones: text-to-text                                           
┏━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃          name ┃         task ┃ output_dim ┃ architecture ┃                                              description ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│  bert-base-en │ text-to-text │        768 │  transformer │         BERT model pre-trained on BookCorpus and English │
│               │              │            │              │                                                Wikipedia │
│ sbert-base-en │ text-to-text │        768 │  transformer │                  Pretrained BERT, fine-tuned on MS Marco │
└───────────────┴──────────────┴────────────┴──────────────┴──────────────────────────────────────────────────────────┘
```

To get a list of supported models during run-time, you can call:

```python
import finetuner

finetuner.describe_models(task='text-to-text')
```

## Image-to-Image Models

For image-to-image tasks, we support two different types of models, `efficientnet` and `resnet`,
both of which having `base` and `large` variations.
- ResNets are suitable for image-to-image search tasks with high performance requirements, where resnet152 is bigger and
  requires higher computational resources than resnet50.
- EfficientNets are more light-weight than ResNet models, making them suitable for image-to-image search tasks
  with low training and inference times.

```bash
                                     Finetuner backbones: image-to-image                                     
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃               name ┃           task ┃ output_dim ┃ architecture ┃                             description ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│  efficientnet-base │ image-to-image │       1792 │          cnn │ EfficientNet B4 pre-trained on ImageNet │
│ efficientnet-large │ image-to-image │       2560 │          cnn │ EfficientNet B7 pre-trained on ImageNet │
│       resnet-large │ image-to-image │       2048 │          cnn │       ResNet152 pre-trained on ImageNet │
│        resnet-base │ image-to-image │       2048 │          cnn │        ResNet50 pre-trained on ImageNet │
└────────────────────┴────────────────┴────────────┴──────────────┴─────────────────────────────────────────┘
```

To get a list of supported models during run-time, you can call:

```python
import finetuner

finetuner.describe_models(task='image-to-image')
```

## Text-to-Image Models

CLIP models are used to text-to-image search tasks, and we support two different types.
- `clip-base-en` and `clip-large-en` are both CLIP models trained only on English text,
  with `clip-large-en` requiring more computational resources than `clip-base-en`.
- `clip-base-multi` is a multilingual model that has been trained on the [laion5b](https://laion.ai/blog/laion-5b/)
  dataset, which contains training from 100+ different langauges. 
  You should use this model when encoding non-English text.

```bash
                                          Finetuner backbones: text-to-image                                           
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃            name ┃          task ┃ output_dim ┃ architecture ┃                                           description ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│    clip-base-en │ text-to-image │        512 │  transformer │                                       CLIP base model │
│   clip-large-en │ text-to-image │       1024 │  transformer │                   CLIP large model with patch size 14 │
│ clip-base-multi │ text-to-image │        512 │  transformer │                                            Open MCLIP │
│                 │               │            │              │  "xlm-roberta-base-ViT-B-32::laion5b_s13b_b90k" model │
└─────────────────┴───────────────┴────────────┴──────────────┴───────────────────────────────────────────────────────┘
```

To get a list of supported models during run-time, you can call:

```python
import finetuner

finetuner.describe_models(task='text-to-image')
```

## Mesh-to-Mesh Models

Currently, we only support one model for mesh-to-mesh search, `pointnet-base`, which we derived from the popular 
[PointNet++ model](https://proceedings.neurips.cc/paper/2017/file/d8bf84be3800d12f74d8b05e9b89836f-Paper.pdf).
The original model is designed for classifying 3D meshes. Our derived model can be used to encode meshes into vectors for search.

```bash
                                        Finetuner backbones: mesh-to-mesh                                         
┏━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃          name ┃         task ┃ output_dim ┃ architecture ┃                                         description ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ pointnet-base │ mesh-to-mesh │        512 │     pointnet │ PointNet++ embedding model for 3D mesh point clouds │
└───────────────┴──────────────┴────────────┴──────────────┴─────────────────────────────────────────────────────┘
```

To get a list of supported models during run-time, you can call:

```python
import finetuner

finetuner.describe_models(task='mesh-to-mesh')
```
