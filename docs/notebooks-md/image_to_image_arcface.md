---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.1
  kernelspec:
    display_name: .venv
    language: python
    name: python3
---

<!-- #region id="p8jc8EyfruKw" -->
# Image-to-Image Search with ArcFaceLoss

<a href="https://colab.research.google.com/drive/1ZS9FmnR9FzO_JYGCPazFM7TcMNQl51xM?usp=sharing"><img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg"></a>

Using image queries to search for visually similar images is a very popular use case. However, pre-trained models do not deliver the best results. Models are trained on general data that lack knowledge related to your specific task. Here's where Finetuner comes in! It enables you to easily add task-specific knowledge to a model.

Where [another guide](https://finetuner.jina.ai/notebooks/image_to_image/) showed off fine-tuning with `TripletMarginLoss`, 
this guide will perform fine-tuning on a dataset with fewer classes, more documents per class and with training data that contains examples from every class in the evaluation data. To improve our performance in this case, we will use `ArcFaceLoss` as our loss function this time.

*Note, please switch to a GPU/TPU Runtime or this will be extremely slow!*

## Install
<!-- #endregion -->

```python id="VdKH0S0FrwS3"
!pip install 'finetuner[full]'
```

<!-- #region id="7EliQdGCsdL0" -->
## Task

We will fine-tune ResNet50 on the [Stanford Cars Dataset](http://ai.stanford.edu/~jkrause/cars/car_dataset.html).
This dataset consists of 196 classes across 16184 documents in total.
Each class represents a single model of car, and consists of roughly 80 pictures of that model of car.

In order to move documents in the same class (images of the same model of car) closer together and move documents of different classes apart, we use the `ArcFaceLoss` function. For more information on how this loss function works, as well as when to use it over `TripletmarginLoss`, see [Advanced Losses and Optimizers](https://finetuner.jina.ai/advanced-topics/advanced-losses-and-optimizers/)

After fine-tuning, documents from each class should have similar embeddings, distinct from documents of other classes, meaning that embedding two images of the same model of car will result in similar output vectors.
<!-- #endregion -->

<!-- #region id="M1sii3xdtD2y" -->
## Data

Our journey starts locally. We have to prepare the data and push it to the Jina AI Cloud and Finetuner will be able to get the dataset by its name. For this example,
we've already prepared the data, and we'll provide Finetuner with just the names of training, query and index dataset (e.g. `stanford-cars-train`).

```{important} 
You don't have to push your data to the Jina AI Cloud before fine-tuning. Instead of a name, you can provide a `DocumentArray` and Finetuner will do upload your data directly.
Important: If your documents refer to locally stored images, please call `doc.load_uri_to_blob()` before starting Finetuner to reduce network transmission and speed up training.
```
<!-- #endregion -->

```python id="L0NfPGbTkNsc"
import finetuner
from finetuner import DocumentArray, Document

finetuner.login(force=True)
```

```python id="ONpXDwFBsqQS"
train_data = DocumentArray.pull('finetuner/stanford-cars-train', show_progress=True)
query_data = DocumentArray.pull('finetuner/stanford-cars-query', show_progress=True)
index_data = DocumentArray.pull('finetuner/stanford-cars-index', show_progress=True)

train_data.summary()
```

<!-- #region id="mUoY1jq0klwk" -->
## Backbone model
Now let's see which backbone models we can use. You can see all the available models by calling `finetuner.describe_models()`.


For this example, we're gonna go with `resnet-base`, a model that has been trained on the [ImageNet](https://www.image-net.org/) classification task. In the next step, Finetuner will adapt this model, turning it into an embedding model instead.
<!-- #endregion -->

<!-- #region id="xA7IIhIOk0h0" -->
## Fine-tuning

Now that we have selected our model and loaded the training and evaluation datasets as `DocumentArray`s, we can start our fine-tuning run.
<!-- #endregion -->

```python id="qGrHfz-2kVC7"
from finetuner.callback import EvaluationCallback

run = finetuner.fit(
    model='resnet-base',
    train_data='finetuner/stanford-cars-train',
    batch_size=128,
    epochs=5,
    learning_rate=1e-3,
    loss='ArcFaceLoss',
    device='cuda',
    sampler='random',
    callbacks=[
        EvaluationCallback(
            query_data='finetuner/stanford-cars-query',
            index_data='finetuner/stanford-cars-index',
        )
    ],
)
```

<!-- #region id="9gvoWipMlG5P" -->
Let's understand what this piece of code does:

* We select a `model`: `resnet-base`.
* We also set `run_name` and `description`, which are optional,
but strongly recommended so that you can access and retain information about your run.
* We specify the training data (`train_data`).
* We set `ArcFaceLoss` as our loss function.
* We use `finetuner.callback.EvaluationCallback` for evaluation and specify the query and index datasets for it. `finetuner/stanford-cars-query` and `finetuner/stanford-cars-index` are two subsamples of the Stanford cars dataset that have no overlap with each other or our training data.
* We set the number of training epochs (`epochs`) and the learning rate (`learning_rate`).
<!-- #endregion -->

<!-- #region id="7ftSOH_olcak" -->
## Monitoring

Now that we've created a run, we can see its status. You can monitor the state of the run with `run.status()`, and use `run.logs()` or `run.stream_logs()` to see the logs.
<!-- #endregion -->

```python id="2k3hTskflI7e"
# note, the fine-tuning might takes 30~ minutes
for entry in run.stream_logs():
    print(entry)
```

<!-- #region id="N8O-Ms_El-lV" -->
Since some runs might take up to several hours, it's important to know how to reconnect to Finetuner and retrieve your runs.

```python
import finetuner
finetuner.login()

run = finetuner.get_run(run.name)
```

You can continue monitoring the runs by checking the status - `finetuner.run.Run.status()` or the logs - `finetuner.run.Run.logs()`. 
<!-- #endregion -->

<!-- #region id="BMpQxydypeZ3" -->
## Evaluating
Currently, we don't have a user-friendly way to get evaluation metrics from the `finetuner.callback.EvaluationCallback` we initialized previously.
What you can do for now is to call `run.logs()` after the end of the run and see the evaluation results:

```bash
Training [5/5] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 48/48 0:00:00 0:00:12 • loss: 13.986
INFO     Done ✨                                                                              __main__.py:195
DEBUG    Finetuning took 0 days, 0 hours 3 minutes and 48 seconds                             __main__.py:197
INFO     Metric: 'resnet_base_precision_at_k' before fine-tuning:  0.11575 after fine-tuning:    __main__.py:210
0.53425
INFO     Metric: 'resnet_base_recall_at_k' before fine-tuning:  0.05745 after fine-tuning:       __main__.py:210
0.27113
INFO     Metric: 'resnet_base_f1_score_at_k' before fine-tuning:  0.07631 after fine-tuning:     __main__.py:210
0.35788
INFO     Metric: 'resnet_base_hit_at_k' before fine-tuning:  0.82900 after fine-tuning: 0.94100  __main__.py:210
INFO     Metric: 'resnet_base_average_precision' before fine-tuning:  0.52305 after fine-tuning: __main__.py:210
0.79779
INFO     Metric: 'resnet_base_reciprocal_rank' before fine-tuning:  0.64909 after fine-tuning:   __main__.py:210
0.89224
INFO     Metric: 'resnet_base_dcg_at_k' before fine-tuning:  1.30710 after fine-tuning: 4.52143  __main__.py:210
INFO     Building the artifact ...                                                            __main__.py:215
INFO     Pushing artifact to Jina AI Cloud ...                                                __main__.py:241
[12:19:53] INFO     Artifact pushed under ID '63f8a9089c6406e19244771d'                                  __main__.py:243
DEBUG    Artifact size is 83.580 MB                                                           __main__.py:245
INFO     Finished 🚀                                                                          __main__.py:246
```
<!-- #endregion -->

<!-- #region id="0l4e4GrspilM" -->
## Saving

After the run has finished successfully, you can download the tuned model on your local machine:

<!-- #endregion -->

```python id="KzfxhqeCmCa8"
artifact = run.save_artifact('resnet-model')
```

<!-- #region id="gkNHTyBkprQ0" -->
## Inference

Now you saved the `artifact` into your host machine,
let's use the fine-tuned model to encode a new `Document`:

```{admonition} Inference with ONNX
In case you set `to_onnx=True` when calling `finetuner.fit` function,
please use `model = finetuner.get_model(artifact, is_onnx=True)`
```
<!-- #endregion -->

```python id="bOi5qcNLplaI"
query = DocumentArray([query_data[0]])

model = finetuner.get_model(artifact=artifact, device='cuda')

finetuner.encode(model=model, data=query)
finetuner.encode(model=model, data=index_data)

assert query.embeddings.shape == (1, 2048)
```

<!-- #region id="1cC46TQ9pw-H" -->
And finally, you can use the embedded `query` to find top-k visually related images within `index_data` as follows:
<!-- #endregion -->

```python id="aYMnyr6ac4ln"
query.match(index_data, limit=10, metric='cosine')
```

<!-- #region id="irvn0igWdLOf" -->
## Before and after
We can directly compare the results of our fine-tuned model with its zero-shot counterpart to get a better idea of how fine-tuning affects the results of a search. Each class of the Stanford cars dataset contains images for a single model of car. Therefore, we can define a 'good' search result as an image of a car that is the same model as the car in the query image, and not necessarily images of cars that are taken at a similar angle, or are the same colour.  
The example below shows exactly this:
<!-- #endregion -->

<!-- #region id="TwL33Jz1datD" -->
query                      |before             |  after
:-------------------------:|:-------------------------:|:-------------------------:
![cars-query](https://user-images.githubusercontent.com/58855099/221186269-a7ebbcd0-6865-45ea-b539-9756d87b3853.png) | ![cars-result-zs](https://user-images.githubusercontent.com/58855099/221186221-6d5bfb9b-2a44-4436-a1af-4c6763eb3b5b.png)  |  ![cars-result-ft](https://user-images.githubusercontent.com/58855099/221187091-adf30d01-9773-4fa6-8e32-b2f45916ff55.png)
![cars-query](https://user-images.githubusercontent.com/58855099/221221384-28734a84-b00a-4605-bfca-28579462ab95.png) | ![cars-result-zs](https://user-images.githubusercontent.com/58855099/221222634-09caec10-6c21-4fba-a098-d9421436d182.png)  |  ![cars-result-ft](https://user-images.githubusercontent.com/58855099/221221342-8a6b1263-f3dd-43d9-bc1f-aa1a7d0cc728.png)

<!-- #endregion -->
