# Finetuning PointConv on PartNet Dataset

In this tutorial we will finetune PointConv on [ModelNet dataset](https://arxiv.org/pdf/1406.5670v3.pdf).
This dataset consists of 573,585 part instances over 26,671 3D models covering 40 object categories.

```{tip}
ModelNet: a consistent, large-scale dataset of 3D objects annotated with fine-grained, instance-level, and hierarchical 3D part information.
This dataset is used to establish three benchmarking tasks for evaluating 3D part recognition: fine-grained semantic segmentation, hierarchical semantic segmentation. In our case we'll use it to finetune PointConv which was pretrained on [ShapeNet dataset](https://arxiv.org/abs/1512.03012).
```

You can download ModelNet dataset on the official [Princeton website](https://modelnet.cs.princeton.edu/).

```shell
wget http://modelnet.cs.princeton.edu/ModelNet40.zip

```

## Data preparation

After downloading ModelNet data we need to unzip it.

```shell
pip install docarray
pip install finetuner
pip install trimesh

unzip ModelNet40.zip
```

Before we go further let's look at some examples from our data:

```{figure} airplane.gif
:align: right
```

```{figure} lap.gif
:align: center
```

```{figure} vase.gif
:align: left
```

```{figure} person.gif
:align: right
```

Now let's go to loading our data. We'll use `DocumentArray` from the library `docarray`. We'll go through all the files in our directory.
And we'll recursively read all files that end with `.off` each file is then loaded into a `Document` and then converted into a point cloud
tensor using `.load_uri_to_point_cloud_tensor` so can also pass the number of point you want, here we pass 2048.

Along with loading point clouds, we also assign the `finetuner_label` as a tag. This label will be later used during finetuning. Finally we save
the resulting `DocumentArray` into a binary file for later use. This way we don't have to convert our data to point clouds again.

```python
from docarray import DocumentArray, Document
import trimesh
import os
from typing import Optional
import glob

train_docs = DocumentArray()
test_docs = DocumentArray()
data_path = 'ModelNet40'
for dir in os.listdir(data_path):
    dir_path = os.path.join(data_path,dir)
    for file in glob.iglob(os.path.join(dir_path,'train/*.off'),recursive=True):
            doc= Document(uri=file)
            doc.load_uri_to_point_cloud_tensor(2048)
            doc.tags['finetuner_label'] = dir
            train_docs.append(doc)
    for file in glob.iglob(os.path.join(dir_path,'test/*.off'),recursive=True):
            doc= Document(uri=file)
            doc.load_uri_to_point_cloud_tensor(2048)
            doc.tags['finetuner_label'] = dir
            test_docs.append(doc)

train_docs.save_binary('train_data_modelnet.bin')
test_docs.save_binary('test_data_modelnet.bin')

```

## Model Training

Now that we have our data ready, we need a model that creates embeddings we will later use those embeddings for searching similar matches for different queries. A lot of work and research has been done in the field of 3D data embeddings there are some powerful models like [PointConv](https://arxiv.org/abs/1811.07246) which have been trained on [ShapeNet dataset](https://arxiv.org/pdf/1512.03012.pdf)

In order to not reivent the wheel we will use this model. We will also use Jina's 3D Mesh Encoder which wraps these two models in an executor.
This executor receives Documents containing point sets data in its blob attribute, with shape (N, 3) and encodes it to embeddings of shape (D,). Now, the following pretrained models are ready to be used to create embeddings:

- PointConv-Shapenet-d512: A PointConv model resulted in 512 dimension of embeddings, which is finetuned based on ShapeNet dataset.
- PointConv-Shapenet-d1024: A PointConv model resulted in 1024 dimension of embeddings, which is finetuned based on ShapeNet dataset.

We import necessary libraries:

```python
import pathlib
from functools import partial

import finetuner
import numpy as np
import torch
from finetuner.tuner.pytorch.losses import TripletLoss
from finetuner.tuner.pytorch.miner import TripletEasyHardMiner
from docarray import Document, DocumentArray

from models import MeshDataModel
```

We define some helper functions for preprocessing and sampling:

```python
def random_sample(pc, num):
    permutation = np.arange(len(pc))
    np.random.shuffle(permutation)
    pc = np.array(pc).astype('float32')
    pc = pc[permutation[:num]]
    return pc



def preprocess(doc: 'Document', num_points: int = 1024, data_aug: bool = True):
    points = random_sample(doc.tensor, num_points)

    points = points - np.expand_dims(np.mean(points, axis=0), 0)  # center
    dist = np.max(np.sqrt(np.sum(points ** 2, axis=1)), 0)
    points = points / dist  # scale

    if data_aug:
        theta = np.random.uniform(0, np.pi * 2)
        rotation_matrix = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )
        points[:, [0, 2]] = points[:, [0, 2]].dot(rotation_matrix)  # random rotation
        points += np.random.normal(0, 0.02, size=points.shape)  # random jitter
    return points
```

Now we'll define some variables:

```python
train_dataset = 'train_data_modelnet.bin' # path for training data
eval_dataset = 'test_data_modelnet.bin' # path for testing data
model_name = 'pointconv' # name of the model you want to use
embed_dim = 512 # dimension for embeddings
batch_size = 64 # how many training instance per batch
epochs = 50 # how many epochs would you like to train for
use_gpu = True # whether we want to use gpu or no
restore_from = False # whether we want to load weights from previously saved model
checkpoint_dir = 'checkpoints' # path where we want to save or load model 
```

In the following code snippet we create MashData model which encapsulates a `PointConv` model with 512 dimensions, we then load
training and evaluation data from the binary files we saved before. We create an optimizer and a learning rate scheduler. In this case with use
an Adam optimizer and MultiStepLR scheduler but you can change those depending on your data and preferences.

```python
model = MeshDataModel(model_name=model_name, embed_dim=embed_dim) # create pointconv model with 512 dimensions
if restore_from: # restore weights from checkpoint
    print(f'==> restore from: {restore_from}')
    ckpt = torch.load(checkpoint_dir, map_location='cpu')
    model.load_state_dict(ckpt)

train_da = DocumentArray.load_binary(train_dataset) # load train dataset
eval_da = DocumentArray.load_binary(eval_dataset) if eval_dataset else None # load eval dataset

def configure_optimizer(model):
    from torch.optim import Adam
    from torch.optim.lr_scheduler import MultiStepLR
    # create Adam optimizer with MultistepLR scheduler
    optimizer = Adam(model.parameters(), lr=5e-4)
    scheduler = MultiStepLR(optimizer, milestones=[30, 60], gamma=0.5)

    return optimizer, scheduler
```

We leverage finetuner's mining strategy to imprive and finetuning the embeddings. By this we mean that the finetuner will use
the mentioned label coming with every `Document` which we specified during preprocessing and will sample it as a positive example
and then sample examples with different label as negatives.

The triplet loss's goal is to pull `Documents` with same class together and those with different classes away from each other thus improving
the final representation.

```python
tuned_model = finetuner.fit(
    model,
    train_da,
    eval_data=eval_da,
    preprocess_fn=partial(preprocess, num_points=2048, data_aug=True),
    epochs=epochs,
    batch_size=batch_size,
    loss=TripletLoss(
        miner=TripletEasyHardMiner(pos_strategy='easy', neg_strategy='semihard')
    ),
    configure_optimizer=configure_optimizer,
    learning_rate=5e-4,
    device='cuda' if use_gpu else 'cpu',
)
# saving the finetuner model
torch.save(
    tuned_model.state_dict(),
    str(checkpoint_dir / f'finetuned-{model_name}-d{embed_dim}.pth'),
)
```

## All in one script

We provide all the code above in one script so it's super easy to use. This script is inside the repository that we clone earlier and here's how you can
use it:

```shell
python executor-3d-encoder/finetune.py --model_name pointconv \
                   --train_dataset ../train_data_modelnet.bin \
                   --eval_dataset ../test_data_modelnet.bin \
                   --batch_size 64 \
                   --epochs 50 \
                   --use-gpu
```

## Evaluating embedding quality

```python
from docarray import DocumentArray
from models import MeshDataModel
import torch
import numpy as np

train_da = DocumentArray.load_binary('../train_data_modelnet.bin') # load train dataset
eval_da = DocumentArray.load_binary('../test_data_modelnet.bin') # load eval dataset

def random_sample(pc, num):
    permutation = np.arange(len(pc))
    np.random.shuffle(permutation)
    pc = np.array(pc).astype('float32')
    pc = pc[permutation[:num]]
    return pc



def preprocess(doc: 'Document', num_points: int = 1024, data_aug: bool = True):
    points = random_sample(doc.tensor, num_points)

    points = points - np.expand_dims(np.mean(points, axis=0), 0)  # center
    dist = np.max(np.sqrt(np.sum(points ** 2, axis=1)), 0)
    points = points / dist  # scale

    if data_aug:
        theta = np.random.uniform(0, np.pi * 2)
        rotation_matrix = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )
        points[:, [0, 2]] = points[:, [0, 2]].dot(rotation_matrix)  # random rotation
        points += np.random.normal(0, 0.02, size=points.shape)  # random jitter
    doc.tensor = points
    return doc

train_da.apply(preprocess)
eval_da.apply(preprocess)

tuned_model = MeshDataModel(model_name='pointconv', embed_dim=512) # create pointconv model with 512 dimensions
tuned_model.load_state_dict(torch.load('checkpoints/finetuned-pointconv-d512.pth'))
tuned_model.eval()

train_da.embed(tuned_model, batch_size=128, device='cuda')
eval_da.embed(tuned_model, batch_size=128, device='cuda')

eval_da.match(train_da, limit=10)
```

### Metric used & results

```python
def hit_rate(da, topk=1):
    hit = 0
    for d in da:
        for m in d.matches[:topk]:
            if d.uri.split('/')[-1].split('_')[0] == m.uri.split('/')[-1].split('_')[0]:
                hit += 1
    return hit/(len(da)*topk)


for k in range(1, 11):
    print(f'hit@{k}:  finetuned: {hit_rate(eval_da, k):.3f}')
```

Now let's findout how does the finetuned model compare to the out of the box model.

The result is demonstrated in the table below:

| hit@k  | pre-trained | fine-tuned |
|--------|-------------|------------|
| hit@1  | 0.068       | 0.122      |
| hit@5  | 0.142       | 0.230      |
| hit@10 | 0.183       | 0.301      |
