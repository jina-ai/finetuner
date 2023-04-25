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

Finding similar 3D Meshes can become very time-consuming. To support this task, one can build search systems. To directly search on the 3D meshes without relying on metadata one can use an encoder model which creates a point cloud from the mesh and encode it into vector dense representations which can be compared to each other. To enable those models to detect the right attributes of a 3D mesh, this tutorial shows you how to use Finetuner to train and use a model for a 3D mesh search system.
<!-- #endregion -->

<!-- #region id="mk4gxLZnYJry" -->
## Install
<!-- #endregion -->

```python colab={"background_save": true} id="vDVkw65kkQcn"
!pip install 'finetuner[full]'
!pip install 'docarray[full]<0.3.0'
```

<!-- #region id="q7Bb9o5ZHSZ3" -->
## Task

Finetuner supports an embedding model which is based on the Pytorch [implementation](https://github.com/yanx27/Pointnet_Pointnet2_pytorch) of the [PointNet++ model](https://proceedings.neurips.cc/paper/2017/file/d8bf84be3800d12f74d8b05e9b89836f-Paper.pdf). This tutorial will show you how to train and use this model for 3D mesh search.

We demonstrate this on the [Modelnet40](https://modelnet.cs.princeton.edu/) dataset, which consists of more than 12k 3D meshes of objects from 40 classes.
Specifically, we want to build a search system, which can receive a 3D mesh and retrieves meshes of the same class.

<!-- #endregion -->

<!-- #region id="H1Yo3NuGP1Oi" -->
## Data

ModelNet40 consists of 9843 meshes provided for training and 2468 meshes for testing. Usually, you would have to download the [dataset](https://modelnet.cs.princeton.edu/) unzip it, [prepare it, and upload it to the Jina AI Cloud](https://finetuner.jina.ai/walkthrough/create-training-data/). After that, you can provide the name of the dataset used for the upload to Finetuner.

For this tutorial, we already prepared the data and uploaded it. Specifically, the training data is uploaded as `modelnet40-train`. For evaluating the model, we split the test set of the original dataset into 300 meshes, which serve as queries (`modelnet40-queries`), and 2168 meshes which serve as the mesh collection, which is searched in (`modelnet40-index`).

Each 3D mesh in the dataset is represented by a [DocArray](https://finetuner.jina.ai/walkthrough/create-training-data/#preparing-a-documentarray) Document object. It contains the URI (local file path) of the original file and a tensor that contains a point cloud with 2048 3D points sampled from the mesh.

```{admonition} Push data to the cloud
We don't require you to push data to the Jina AI Cloud by yourself. Instead of a name, you can provide a `DocumentArray` or a path to a CSV file.
In those cases Finetuner will do the job for you.
When you construct a DocArray dataset with documents of 3D meshes, please call `doc.load_uri_to_point_cloud_tensor(2048)` to create point clouds from your local mesh files before pushing the data to the cloud since Finetuner has no access to your local files.
```

The code below loads the data and prints a summary of the training datasets:
<!-- #endregion -->

```python id="uTDreSwfYGOR"
import finetuner
from finetuner import DocumentArray, Document

finetuner.login(force=True)
```

```python id="Y-Um5gE8IORv"
train_data = DocumentArray.pull('finetuner/modelnet40-train', show_progress=True)
query_data = DocumentArray.pull('finetuner/modelnet40-queries', show_progress=True)
index_data = DocumentArray.pull('finetuner/modelnet40-index', show_progress=True)

train_data.summary()
```

<!-- #region id="r4cP95RzLybw" -->
Now, we want to take a look at the point clouds of some of the meshes. Therefore, you can use the `display` function:
<!-- #endregion -->

```python id="kCv455NPMD1O"
index_data[0].display()
```

<!-- #region id="XlttkaD5Omhk" -->
![A point cloud example](https://user-images.githubusercontent.com/6599259/208113813-bcf498d9-edf7-4496-a087-03bb783f3b70.png)
<!-- #endregion -->

<!-- #region id="B3I_QUeFT_V0" -->
## Backbone model

The model we provide for 3d mesh encoding is called `pointnet-base`. In the following, we show you how to train it on the ModelNet training dataset.
<!-- #endregion -->

<!-- #region id="lqg0eY9oknLL" -->
## Fine-tuning

Now that we have data for training and evaluation. as well as the name of the model, which we want to train, we can configure and submit a fine-tuning run:
<!-- #endregion -->

```python id="rR22MbgITp8M"
from finetuner.callback import EvaluationCallback

run = finetuner.fit(
    model='pointnet-base',
    train_data='finetuner/modelnet40-train',
    epochs=10,
    batch_size=64,
    learning_rate= 5e-4,
    loss='TripletMarginLoss',
    device='cuda',
    callbacks=[
        EvaluationCallback(
            query_data='finetuner/modelnet40-queries',
            index_data='finetuner/modelnet40-index',
            batch_size=64,
        )
    ],
)
```

<!-- #region id="ossT9LH1oh6K" -->
Let's understand what this piece of code does:

* We start with providing a `model` name, in our case "pointnet-base".
* Via the `train_data` parameter, we inform the Finetuner about the name of the dataset in the Jina AI Cloud
* We also provide some hyper-parameters such as the number of `epochs`, `batch_size`, and a `learning_rate`.
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

You can continue monitoring the run by checking the status - `finetuner.run.Run.status()` or the logs - `finetuner.run.Run.logs()`.
<!-- #endregion -->

<!-- #region id="WgTrq9D5q0zc" -->
## Evaluating

Our `EvaluationCallback` during fine-tuning ensures that after each epoch, an evaluation of our model is run. We can access the results of the last evaluation in the logs as follows `print(run.logs())`:

```bash
  Training [10/10] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 154/154 0:00:00 0:00:26 • loss: 0.001
           INFO     Done ✨                                                                                                                                            __main__.py:195
           DEBUG    Finetuning took 0 days, 0 hours 5 minutes and 39 seconds                                                                                           __main__.py:197
           INFO     Metric: 'pointnet_base_precision_at_k' before fine-tuning:  0.56533 after fine-tuning: 0.81100                                                        __main__.py:210
           INFO     Metric: 'pointnet_base_recall_at_k' before fine-tuning:  0.15467 after fine-tuning: 0.24175                                                           __main__.py:210
           INFO     Metric: 'pointnet_base_f1_score_at_k' before fine-tuning:  0.23209 after fine-tuning: 0.34774                                                         __main__.py:210
           INFO     Metric: 'pointnet_base_hit_at_k' before fine-tuning:  0.95667 after fine-tuning: 0.95333                                                              __main__.py:210
           INFO     Metric: 'pointnet_base_average_precision' before fine-tuning:  0.71027 after fine-tuning: 0.85515                                                     __main__.py:210
           INFO     Metric: 'pointnet_base_reciprocal_rank' before fine-tuning:  0.79103 after fine-tuning: 0.89103                                                       __main__.py:210
           INFO     Metric: 'pointnet_base_dcg_at_k' before fine-tuning:  4.71826 after fine-tuning: 6.41999                                                              __main__.py:210
           INFO     Building the artifact ...                                                                                                                          __main__.py:215
           INFO     Saving artifact locally ...                                                                                                                        __main__.py:237
[15:46:55] INFO     Artifact saved in artifacts/                                                                                                                       __main__.py:239
           DEBUG    Artifact size is 27.379 MB                                                                                                                         __main__.py:245
           INFO     Finished 🚀                                                                                                                                        __main__.py:246

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
<!-- #endregion -->

```python id="rDGxi7kVq_sH"
model = finetuner.get_model(artifact=artifact, device='cuda')

finetuner.encode(model=model, data=query_data)
finetuner.encode(model=model, data=index_data)

assert query.embeddings.shape == (1, 512)
```

<!-- #region id="pfoc4YG4rrkI" -->
And finally, you can use the embedded `query` to find top-k visually related images within `index_data` as follows:
<!-- #endregion -->

```python id="_jGsSyedrsJp"
query_data.match(index_data, limit=10, metric='cosine')
```

<!-- #region id="xGAjr26o6j-n" -->
To compare the matches against results obtained with a pointnet-base model without training, you can use the `build_model` function:
<!-- #endregion -->

```python id="cChTjw3b6iXq"
zero_shot_model = finetuner.build_model('pointnet-base')

finetuner.encode(model=zero_shot_model, data=query_data)
finetuner.encode(model=zero_shot_model, data=index_data)

query_data.match(index_data, limit=10, metric='cosine')
```

<!-- #region id="CgZHPInNWWHn" -->
## Before and After

After the inference, you can investigate the results with the `display` function, as shown in the code block below:
<!-- #endregion -->

```python id="p37Ryip2dKoO"
query_data[5].display()
query_data[5].matches[0].display()
```

<!-- #region id="lybyfx6OdMJL" -->
While you will notice that the PointNet++ might already deliver good results for some queries without training, the fine-tuned model does perform better on many queries like the ones shown below:

![Results](https://user-images.githubusercontent.com/6599259/208681224-aa3263f2-326a-4c66-baf0-7fa1dbf594a2.png)


<!-- #endregion -->
