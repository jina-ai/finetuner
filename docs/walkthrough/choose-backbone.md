(choose-backbone)=
# Backbone Model

Finetuner provides several widely used backbone models,
including `resnet`, `efficientnet`, `clip`, `bert` and `pointnet`.
Each model has its own variants and is designed to produce embeddings for a specific modality.

Finetuner will convert these backbone models to embedding models by removing
the *head* or applying *pooling*,
performing fine-tuning and producing the final embedding model.
The embedding model can be fine-tuned for text-to-text, image-to-image, text-to-image or mesh-to-mesh
search tasks.

## Text-to-Text Models

We support two different variations of the `bert` model for text-to-text encoding tasks.
- `bert-base-en` is a pre-trained model that has undergone no finetuning.
  It should be used if you want to train something from scratch or train a model on a task where a
  notion of similarity you want to finetune on is very different
  from what's usually considered as similar.
- `sbert-base-en` has been fine-tuned once by [sentence-transformers](https://www.sbert.net/) on the 
  [MS MARCO](https://microsoft.github.io/msmarco/) dataset on top of BERT.
  It is designed for matching web search queries to short text passages and 
  is a suitable backbone for traditional text-to-text search tasks.

These transformer-based models are loaded using the huggingface
[transformers](https://github.com/huggingface/transformers) library.
It is also worth mentioning that these two models have the same `output_dim`, meaning that the shape of
the embeddings produced are the same.
This means that while the performance of these models may differ depending on the domain, they can be used interchangeably.

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

To get a list of supported models at run-time, you can call:

```python
import finetuner

finetuner.describe_models(task='text-to-text')
```

## Image-to-Image Models

For image-to-image tasks, we support two different types of models, `efficientnet` and `resnet`,
both of which having `base` and `large` variations.
- ResNets are suitable for image-to-image search tasks with high performance requirements, where resnet152 (large variant)
  is bigger and requires higher computational resources than resnet50 (base variant).
- EfficientNets are more light-weight than ResNet models, making them suitable for image-to-image search tasks
  with low training and inference times.

ResNet/EfficientNet models are loaded using the [torchvision](https://pytorch.org/vision/stable/index.html) library.

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

To get a list of supported models at run-time, you can call:

```python
import finetuner

finetuner.describe_models(task='image-to-image')
```

## Text-to-Image Models

CLIP models are used for text-to-image search tasks. We support two different types of CLIP models:
- English-only models `clip-base-en` and `clip-large-en`,
  with `clip-large-en` requiring more computational resources than `clip-base-en`.
- A multilingual model `clip-base-multi`, trained on the [laion5b](https://laion.ai/blog/laion-5b/)
  dataset, which contains data from 100+ different langauges. 
  You should use this model when encoding non-English text.

Like the models used for text-to-text tasks, these models are transformer-based, and are loaded using the huggingface
[transformers](https://github.com/huggingface/transformers) library.

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

## Model Options

Once you have chosen a model you can provide a dictionary of options to adjust the behaviour of the model to the
`model_options` parameter of the {meth}`finetuner.fit` function.
You can see a list of additional options that you can provide for each model using the
{meth}`finetuner.list_model_options` function:

![get_model_options](../imgs/get_model_options.png)

While each model has their own set of options, the `sbert-base-en` model, as shown in the example above, has the four most common options.
### `collate_options` and `preprocess_options`
Collation and preprocessing are part of data preparation step of finetuning. Typically, these default parameters for these two functions perform well, so these options do not need to be adjusted.

### `pooler` and `pooler_options`

Pooling layers are layers in a machine learning model that are used to reduce the dimensionality of data. This is usually done for one of two reasons: to remove unnecessary information contained within an embedding of a larger size, or when a model outputs multiple embeddings and only one embedding is needed. Typically, there are two ways to do this: average pooling or max pooling.
While a model may have many pooling layers within it, it is unwise to replace a pooling layer with another unless it is the last layer of the model.  
In cases where your chosen model does have a pooling layer as its last layer, Finetuner allows you to replace the default pooler with a `GeM` pooling layer.
Currently, all of our [text-to-text](#text-to-text-models) and [image-to-image](#image-to-image-models)
models support replacing the pooling layer.

### GeM pooling

`GeM` (Generalised Mean) pooling is an advanced pooling technique that is popular for computer vision and face recognition tasks.  
The `GeM` pooler has two adjustable parameters: a scaling parameter `p` and an epsilon `eps`.
At `p = 1`, the `GeM` pooler will act like an average pooler.
As `p` increases, more weight is given to larger values, making it act more like max pooling.
`eps` is used to clamp values to be slightly above 0, and altering this won't result in much change to the performance.
By default, `p=3` and `eps=1e-6`. You can specify the pooler and adjust these parameters in a dictionary provided to the `model_options` parameter:
```python
run = finetuner.fit(
    ...,
    model_options = {
        ...
        'pooler': 'GeM',
        'pooler_options': {'p': 2.4, 'eps': 1e-5}
    }
)
```

## Models for Data Synthesis

When creating data synthesis jobs, two different types of models need to be chosen, the `relation_miner`
and the `cross_encoder`.
These are passed to the {meth}`finetuner.synthesize` function in a {class}`~finetuner.data.SynthesisModels` object.

```python
from finetuner import synthesize
from finetuner.data import SynthesisModels

synthesize(
    query_data='my_query_data',
    corpus_data='my_corpus_data',
    models=SynthesisModels(
      'sbert-base-en',
      'crossencoder-base-en',
    ),
)

```
To use the recommended `relation_miner` and `cross_encoder` models, you can pass the
`finetuner.data.DATA_SYNTHESIS_EN` constant, which is a premade {class}`~finetuner.data.SynthesisModels` object.

```python
from finetuner import synthesize
from finetuner.data import DATA_SYNTHESIS_EN

synthesize(
    query_data='my_query_data',
    corpus_data='my_corpus_data',
    models=DATA_SYNTHESIS_EN,
)

```


The `relation_miner` model can be any [text-to-text](#text-to-text-models) model, though we strongly encourage you to use 
`sbert-base-en`, as `bert-base-en` will not perform as well.
Currently, we only support one cross-encoder model, `crossencoder-base-en`.