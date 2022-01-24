# Finetuning PointConv on PartNet Dataset

In this tutorial we will finetune PointConv on [ModelNet40 dataset](https://arxiv.org/pdf/1406.5670v3.pdf).
The ModelNet40 dataset contains synthetic object point clouds. As the most widely used benchmark for point cloud analysis, ModelNet40 is popular because of its various categories, clean shapes, well-constructed dataset.

The original ModelNet40 consists of 12,311 CAD-generated meshes in 40 categories (such as airplane, car, plant, lamp), of which 9,843 are used for training while the rest 2,468 are reserved for testing.

```{tip}
ModelNet: a consistent, large-scale dataset of 3D objects annotated with fine-grained, instance-level, and hierarchical 3D part information.
The corresponding point cloud data points are uniformly sampled from the mesh surfaces, and then further preprocessed by moving to the origin and scaling into a unit sphere. This dataset is used to establish three benchmarking tasks for evaluating 3D part recognition: fine-grained semantic segmentation and classification. In our case we'll use it to finetune PointConv which was pretrained on [ShapeNet dataset](https://arxiv.org/abs/1512.03012).
```

You can download ModelNet dataset on the official [Princeton website](https://modelnet.cs.princeton.edu/).

```shell
wget http://modelnet.cs.princeton.edu/ModelNet40.zip

```

## Data preparation

After downloading ModelNet data we need to unzip it.

```shell
pip install finetuner
pip install trimesh

unzip ModelNet40.zip
```

Before we go further let's look at some examples from our data:

````{dropdown} How to visualize 3D data?

```shell
pip install plotly
pip install numpy
pip install matplotlib
pip install scipy
```

```python
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import scipy.spatial.distance

def read_off(file):
    off_header = file.readline().strip()
    if 'OFF' == off_header:
        n_verts, n_faces, __ = tuple([int(s) for s in file.readline().strip().split(' ')])
    else:
        n_verts, n_faces, __ = tuple([int(s) for s in off_header[3:].split(' ')])
    verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
    faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
    return verts, faces


def visualize_rotate(data):
    x_eye, y_eye, z_eye = 1.25, 1.25, 0.8
    frames=[]

    def rotate_z(x, y, z, theta):
        w = x+1j*y
        return np.real(np.exp(1j*theta)*w), np.imag(np.exp(1j*theta)*w), z

    for t in np.arange(0, 10.26, 0.1):
        xe, ye, ze = rotate_z(x_eye, y_eye, z_eye, -t)
        frames.append(dict(layout=dict(scene=dict(camera=dict(eye=dict(x=xe, y=ye, z=ze))))))
    fig = go.Figure(data=data,
        layout=go.Layout(
            updatemenus=[dict(type='buttons',
                showactive=False,
                y=1,
                x=0.8,
                xanchor='left',
                yanchor='bottom',
                pad=dict(t=45, r=10),
                buttons=[dict(label='Play',
                    method='animate',
                    args=[None, dict(frame=dict(duration=50, redraw=True),
                        transition=dict(duration=0),
                        fromcurrent=True,
                        mode='immediate'
                        )]
                    )
                ])]
        ),
        frames=frames
    )

    return fig


def pcshow(xs,ys,zs):
    data=[go.Scatter3d(x=xs, y=ys, z=zs,
                                   mode='markers')]
    fig = visualize_rotate(data)
    fig.update_traces(marker=dict(size=2,
                      line=dict(width=2,
                      color='DarkSlateGrey')),
                      selector=dict(mode='markers'))
    fig.show()

def visualize(uri):
    with open(uri, 'r') as f:
        verts, faces = read_off(f)
        
    i,j,k = np.array(faces).T
    x,y,z = np.array(verts).T
    visualize_rotate([go.Mesh3d(x=x, y=y, z=z, color='gray', opacity=0.5, i=i,j=j,k=k)]).show()
```
````

```{figure} airplane.gif
:align: center
:width: 70%
```

```{figure} lap.gif
:align: center
:width: 70%
```

```{figure} vase.gif
:align: center
:width: 70%
```

```{figure} person.gif
:align: center
:width: 70%
```

Now let's go to loading our data. We'll use `DocumentArray` from the library `docarray`. We'll go through all the files in our directory.
And we'll recursively read all files that end with `.off` each file is then loaded into a `Document` and then converted into a point cloud
tensor using `.load_uri_to_point_cloud_tensor` so can also pass the number of point you want, here we pass 2048.

Along with loading point clouds, we also assign the `finetuner_label` as a tag. This label will be later used during finetuning. Finally we save
the resulting `DocumentArray` into a binary file for later use. This way we don't have to convert our data to point clouds again.

```python
import glob
import os
from typing import Optional

import trimesh
from docarray import Document, DocumentArray

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

import numpy as np
import torch
from docarray import Document, DocumentArray
from models import MeshDataModel

import finetuner
from finetuner.tuner.pytorch.losses import TripletLoss
from finetuner.tuner.pytorch.miner import TripletEasyHardMiner
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

As we already have an Encoder that uses `PointConv` to embed data let's use it.  

```shell
git clone https://github.com/jina-ai/executor-3d-encoder.git
```

In the following code snippet we create MeshData model which encapsulates a `PointConv` model with 512 dimensions, we then load
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

## Evaluating embedding quality & Results

Now let's see whether we made an improvement or not. We will use the training data as index and the testing data as query. We'll embed them both
and then search for test 3d objects (the query) in our training data (index). We do this using the pretrained model and the finetuned model and then compare top matches.

We'll use two metrics to evaluate our two models (pretrained, straight out of the box model and finetuned model) on the search task:

**mAP@k** : We'll calculate the average precision at 1, 5 and 10, then we'll calculate the mean of those average precisions for all documents in our test data (these are our queries) because we care about how accurate and precise our retrieved 3D objects are.
**mNDCG@k**: We'll calculate NDCG at 1,5 and 10, then we'll calculate the mean of those for all documents in our test data  because we care about the order of the retrieved 3D objects.

````{dropdown} Complete source code

```python
train_da = DocumentArray.load_binary('../train_data_modelnet.bin') # load train dataset
eval_da = DocumentArray.load_binary('../test_data_modelnet.bin') # load test dataset

train_da.apply(preprocess)
eval_da.apply(preprocess)

tuned_model = MeshDataModel(model_name='pointconv', embed_dim=512) # create pointconv model with 512 dimensions
tuned_model.load_state_dict(torch.load('checkpoints/finetuned-pointconv-d512.pth')) # load finetuned weights
tuned_model.eval()

train_da.embed(tuned_model, batch_size=128, device='cuda') # create embeddings
eval_da.embed(tuned_model, batch_size=128, device='cuda')

eval_da.match(train_da, limit=10)

def mean_average_precision_at_k(da, topk=1):
    hit = []
    avg_pr = []
    for d in da:
        for m in d.matches[:topk]:
            if d.uri.split('/')[-1].split('_')[0] == m.uri.split('/')[-1].split('_')[0]:
                hit.append(1)
            else:
                hit.append(0)
        avg_pr.append(average_precision(hit))
    return np.mean(avg_pr)


for k in range(1, 11):
    print(f'mAP@{k}:  finetuned: {mean_average_precision_at_k(eval_da, k):.3f}')
```
````

The difference is shown in the tables below:

| mAP@k  | pre-trained | fine-tuned |
|--------|-------------|------------|
| mAP@1  | 0.147       | 0.719      |
| mAP@5  | 0.113       | 0.697      |
| mAP@10 | 0.100       | 0.686      |

| mNDCG@k  | pre-trained | fine-tuned |
|--------|-------------|--------------|
| mNDCG@1  | 0.563       | 0.927      |
| mNDCG@5  | 0.617       | 0.931      |
| mNDCG@10 | 0.647       | 0.935      |

Now let's do some queries ourselves and check the visualizations

````{dropdown} Complete source code

```python
train_da = DocumentArray.load_binary('../train_data_modelnet.bin') # load train dataset
eval_da = DocumentArray.load_binary('../test_data_modelnet.bin') # load eval dataset

train_da.apply(preprocess)
eval_da.apply(preprocess)

tuned_model = MeshDataModel(model_name='pointconv', embed_dim=512) # create pointconv model with 512 dimensions
tuned_model.load_state_dict(torch.load('checkpoints/finetuned-pointconv-d512.pth'))
tuned_model.eval()

train_da.embed(tuned_model, batch_size=128, device='cuda')
eval_da.embed(tuned_model, batch_size=128, device='cuda')

eval_da.match(train_da, limit=10)
visualize(eval_da.matches[0].uri)
```
````

Query:

```{figure} radio_query.gif
:align: center
:width: 70%
```

Pretrained PointConv:

```{figure} radio_oob.gif
:align: center
:width: 70%
```

Finetuned PointConv:

```{figure} radio_finetuned.gif
:align: center
:width: 70%
```

Query:

```{figure} dresser_query.gif
:align: center
:width: 70%
```

Pretrained PointConv:

```{figure} dresser_oob.gif
:align: center
:width: 70%
```

Finetuned PointConv:

```{figure} dresser_finetuned.gif
:align: center
:width: 70%
```

We can clearly see that after finetuning PointConv has enhanced embeddings, and that's how you finetune a 3D model.
