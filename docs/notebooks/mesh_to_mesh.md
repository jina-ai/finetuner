---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.1
  kernelspec:
    display_name: Python 3
    name: python3
---

<!-- #region id="C0RxIJmLkTGk" -->
# 3D Mesh-to-3D Mesh Search via PointNet++

<a href="https://colab.research.google.com/drive/1lIMDFkUVsWMshU-akJ_hwzBfJ37zLFzU?usp=sharing"><img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg"></a>

Finding similar 3D Meshes can become very time consuming. To support this task, one can build search systems. To directly search on the 3D meshes without relying on metadata one can use encoder model which extract create a point cloud from the mesh and encode it into vector dense representations which can be compared to each other. To enable those models to detect the right attributes of an 3D Mesh, this tutorial show you how to use Finetuner to train and use a model for 3D mesh search system.
<!-- #endregion -->

<!-- #region id="mk4gxLZnYJry" -->
## Install
<!-- #endregion -->

```python id="vDVkw65kkQcn"
!pip install 'finetuner[full]'
```

<!-- #region id="q7Bb9o5ZHSZ3" -->
## Task

Finetuner supports an embedding model which is based on the Pytorch [implemention](https://github.com/yanx27/Pointnet_Pointnet2_pytorch) of the [PointNet++ model](https://proceedings.neurips.cc/paper/2017/file/d8bf84be3800d12f74d8b05e9b89836f-Paper.pdf). This tutorial will show you how to train and use this model for 3D mesh search.

We demonstrate this on the [Modelnet40](https://modelnet.cs.princeton.edu/) dataset which consist of more than 12k 3D meshes of objects from 40 classes.
Specifically, we want to build a search system, which can receive a 3D mehs and retrieves meshes of the same class.

We will buid a dataset with some images for 

<!-- #endregion -->

<!-- #region id="H1Yo3NuGP1Oi" -->
## Data

ModelNet40 consists of 9843 meshes provided for training and 2468 meshes for testing. Usually, you would have to download the [dataset](https://modelnet.cs.princeton.edu/) unzip it, [prepare it, and upload it to the Jina AI Cloud](https://https://finetuner.jina.ai/walkthrough/create-training-data/). After that, you can provide the name of the dataset used for the upload to Finetuner.

For this tutorial, we already prepared the data and uploaded it. Specifically the training data is uploaded as `modelnet40-train`. For evaluating the model, we split the test set of the original dataset in 300 meshes, which serve as queries (`modelnet40-queries`) and 2168 meshes which serve as the mesh collection, which is searched in (`modelnet40-index`).

Each 3D mesh in the dataset is represented by a [DocArray](https://github.com/docarray/docarray) Document object. It contains the uri (local filepath) of the original file and a tensor which contains a point cloud with 2048 3D points sampled from the mesh as explained in (TODO add link to documentation)

```{admonition} Push data to the cloud
We don't require you to push data to the Jina AI Cloud by yourself. Instead of a name, you can provide a `DocumentArray` or a path to a CSV file.
In those cases Finetuner will do the job for you.
When you construct a DocArray dataset with documents of 3D meshes, please call `doc.load_uri_to_point_cloud_tensor(2048)` to create point clouds from your local mesh files before pushing the data to the cloud since Finetuner has no access to your local files.
```

The code below loads the data and prints a summary of the training datasets:
<!-- #endregion -->

```python id="uTDreSwfYGOR"
import finetuner
from docarray import DocumentArray, Document

finetuner.login(force=True)
```

```python id="Y-Um5gE8IORv"
train_data = DocumentArray.pull('modelnet40-train', show_progress=True)
query_data = DocumentArray.pull('modelnet40-queries', show_progress=True)
index_data = DocumentArray.pull('modelnet40-index', show_progress=True)

train_data.summary()
```

<!-- #region id="B3I_QUeFT_V0" -->
## Backbone model

The model, we provide for 3d mesh encoding is called `pointnet++`. In the following, we show you how to train it on the modelnet training dataset.
<!-- #endregion -->

<!-- #region id="lqg0eY9oknLL" -->
## Fine-tuning

Now that we have data for training and evaluation as well as the name of the model, which we want to train, we can configure and submit a fine-tuning run:
<!-- #endregion -->

```python id="rR22MbgITp8M"
from finetuner.callback import EvaluationCallback

run = finetuner.fit(
    model='pointnet++',
    train_data='modelnet40-train',
    epochs=5,
    batch_size=64,
    learning_rate= 5e-4,
    loss='TripletMarginLoss',
    device='cpu',
    callbacks=[
        EvaluationCallback(
            query_data='modelnet40-queries',
            index_data='modelnet40-index',
            batch_size=64,
        )
    ],
)
```

<!-- #region id="ossT9LH1oh6K" -->
Let's understand what this piece of code does:

* We start with providing `model`, in our case "pointnet++".
* Via the `train_data` parameter, we inform the Finetuner about the name of the dataset in the Jina AI Cloud
* We also provide some hyper-parameters such as number of `epochs`, `batch_size`, and a `learning_rate`.
* We use `TripletMarginLoss` to optimize the PointNet++ model.
* We use an evaluation callback, which uses the fine-tuned model for encoding the text queries and meshes in the index data collection. It also accepts the `batch_size` attribute. By encoding 64 meshes at once, the evaluation gets faster.

<!-- #endregion -->

<!-- #region id="AsHsMJP6p7Co" -->
## Monitoring

Now that we've created a run, let's see how it's processing. You can monitor the run by checking the status via `run.status()` and view the logs with `run.logs()`. To stream logs, call `run.stream_logs()`:
<!-- #endregion -->

```python id="PCCRZ6PalsK3"
# note, the fine-tuning might takes 20~ minutes
for entry in run.stream_logs():
    print(entry)
```

<!-- #region id="zG7Uci-qqkzM" -->
Since some runs might take up to several hours/days, it's important to know how to reconnect to Finetuner and retrieve your run.

```python
import finetuner

finetuner.login()
run = finetuner.get_run(run.name)
```

You can continue monitoring the run by checking the status - `finetuner.run.Run.status()` or the logs - `finetuner.run.Run.logs()`.*kursiver Text*
<!-- #endregion -->

<!-- #region id="WgTrq9D5q0zc" -->
## Evaluating

Our `EvaluationCallback` during fine-tuning ensures that after each epoch, an evaluation of our model is run. We can access the results of the last evaluation in the logs as follows `print(run.logs())`:

```bash
...
...
```

<!-- #endregion -->

<!-- #region id="W4ZCKUOfq9oC" -->

After the run has finished successfully, you can download the tuned model on your local machine:
<!-- #endregion -->

```python id="K5UdKleiqd8m"
artifact = run.save_artifact('pointnet_model')
```

<!-- #region id="JU3uUVyirTE1" -->
## Inference

Now you saved the `artifact` into your host machine,
let's use the fine-tuned model to encode a new `Document`:

```{admonition} Inference with ONNX
In case you set `to_onnx=True` when calling `finetuner.fit` function,
please use `model = finetuner.get_model(artifact, is_onnx=True)`
```
<!-- #endregion -->

```python id="rDGxi7kVq_sH"
query = DocumentArray([query_data[0]])

model = finetuner.get_model(artifact=artifact, device='cuda')

finetuner.encode(model=model, data=query)
finetuner.encode(model=model, data=index_data)

assert query.embeddings.shape == (1, 512)
```

<!-- #region id="pfoc4YG4rrkI" -->
And finally you can use the embeded `query` to find top-k visually related images within `index_data` as follows:
<!-- #endregion -->

```python id="_jGsSyedrsJp"
query.match(index_data, limit=10, metric='cosine')
```
