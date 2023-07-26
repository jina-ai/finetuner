(choose-backbone)=
# Backbone Model

Finetuner provides several widely used backbone models,
including `resnet`, `efficientnet`, `clip` and `bert`.
Thereby, for most of them, Finetuner provides multiple variants, e.g., the common `resnet50 ` and the more complex `resnet152` model.

Finetuner will convert these backbone models to embedding models by removing
the *head* or applying *pooling*,
performing fine-tuning and producing the final embedding model.
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
````{tab} mesh-to-mesh
```python
import finetuner

finetuner.describe_models(task='mesh-to-mesh')
```
````

to get a list of supported models:

````{tab} text-to-text
```bash
                                                       Finetuner backbones: text-to-text                                                       
┏━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                   name ┃         task ┃ output_dim ┃ architecture ┃                                                             description ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ jina-embedding-t-en-v1 │ text-to-text │        312 │  transformer │    Text embedding model trained using Linnaeus-Clean dataset by Jina AI │
│ jina-embedding-s-en-v1 │ text-to-text │        512 │  transformer │    Text embedding model trained using Linnaeus-Clean dataset by Jina AI │
│ jina-embedding-b-en-v1 │ text-to-text │        768 │  transformer │    Text embedding model trained using Linnaeus-Clean dataset by Jina AI │
│ jina-embedding-l-en-v1 │ text-to-text │       1024 │  transformer │    Text embedding model trained using Linnaeus-Clean dataset by Jina AI │
│           bert-base-en │ text-to-text │        768 │  transformer │              BERT model pre-trained on BookCorpus and English Wikipedia │
│        bert-base-multi │ text-to-text │        768 │  transformer │                        BERT model pre-trained on multilingual Wikipedia │
│   distiluse-base-multi │ text-to-text │        512 │  transformer │      Knowledge distilled version of the multilingual Sentence Encoder   │
│          sbert-base-en │ text-to-text │        768 │  transformer │                                 Pretrained BERT, fine-tuned on MS Marco │
└────────────────────────┴──────────────┴────────────┴──────────────┴─────────────────────────────────────────────────────────────────────────┘
```
````
````{tab} image-to-image
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
````
````{tab} text-to-image
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
````
````{tab} mesh-to-mesh
```bash
                                        Finetuner backbones: mesh-to-mesh                                         
┏━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃          name ┃         task ┃ output_dim ┃ architecture ┃                                         description ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ pointnet-base │ mesh-to-mesh │        512 │     pointnet │ PointNet++ embedding model for 3D mesh point clouds │
└───────────────┴──────────────┴────────────┴──────────────┴─────────────────────────────────────────────────────┘
```
````

+ ResNets are suitable for image-to-image search tasks with high performance requirements, where `resnet152` is bigger and requires higher computational resources than `resnet50`.
+ EfficientNets are suitable for image-to-image search tasks with low training and inference times. The model is more light-weighted than ResNet. Here, `efficientnet_b4` is the bigger and more complex model.
+ CLIP is the one for text-to-image search, where the images do not need to have any text descriptors.
+ BERT is generally suitable for text-to-text search tasks.
+ Msmarco-distilbert-base-v3 is designed for matching web search queries to short text passages and is a suitable backbone for similar text-to-text search tasks.
+ PointNet++ is an embedding model, which we derived from the popular [PointNet++ model](https://proceedings.neurips.cc/paper/2017/file/d8bf84be3800d12f74d8b05e9b89836f-Paper.pdf).
  The original model is designed for classifying 3D meshes. Our derived model can be used to encode meshes into vectors for search.

It should be noted that:

+ ResNet/EfficientNet models are loaded from the [torchvision](https://pytorch.org/vision/stable/index.html) library.
+ Transformer-based models are loaded from the huggingface [transformers](https://github.com/huggingface/transformers) library.
+ `msmarco-distilbert-base-v3` has been fine-tuned once by [sentence-transformers](https://www.sbert.net/) on the [MS MARCO](https://microsoft.github.io/msmarco/) dataset on top of BERT.