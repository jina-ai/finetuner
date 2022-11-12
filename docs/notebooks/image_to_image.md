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

<!-- #region id="p8jc8EyfruKw" -->
# Image-to-Image Search via ResNet50

<a href="https://colab.research.google.com/drive/1QuUTy3iVR-kTPljkwplKYaJ-NTCgPEc_?usp=sharing"><img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg"></a>

Searching visually similar images with image queries is a very popular use-case. However, using pre-trained models does not deliver the best results – the models are trained on general data that lack the particularities of your specific task. Here's where Finetuner comes in! It enables you to accomplish this easily.

This guide will demonstrate how to fine-tune a ResNet model for image to image retrieval.

*Note, please consider switching to GPU/TPU Runtime for faster inference.*

## Install
<!-- #endregion -->

```python id="VdKH0S0FrwS3"
!pip install 'finetuner[full]'
```

<!-- #region id="7EliQdGCsdL0" -->
## Task

More specifically, we will fine-tune ResNet50 on [Totally Looks Like Dataset](https://sites.google.com/view/totally-looks-like-dataset).
The dataset consists of 6016 pairs of images (12032 in total).

The dataset consists of pairs of images, these are the positive pairs. Negative pairs are constructed by taking two different images, i.e. images that are not in the same pair initially. Following this approach, we construct triplets and use the `TripletLoss`. You can find more in the [how Finetuner works](https://finetuner.jina.ai/get-started/how-it-works/#contrastive-metric-learning) section.

After fine-tuning, the embeddings of positive pairs are expected to be pulled closer, while the embeddings for negative pairs are expected to be pushed away.
<!-- #endregion -->

<!-- #region id="M1sii3xdtD2y" -->
## Data

Our journey starts locally. We have to prepare the data and push it to the Jina AI Cloud and Finetuner will be able to get the dataset by its name. For this example,
we already prepared the data, and we'll provide the names of training data (`tll-train-data`) directly to Finetuner.

```{important} 
We don't require you to push data to the Jina AI Cloud by yourself. Instead of a name, you can provide a `DocumentArray` and Finetuner will do the job for you.
```

```{important}
When working with Document where images are stored locally, please call `doc.load_uri_to_image_tensor(width=224, height=224)` or another image shape to reduce network transmission and speed up training.
```
<!-- #endregion -->

```python id="L0NfPGbTkNsc"
import finetuner
from docarray import DocumentArray, Document

finetuner.login(force=True)
```

```python id="ONpXDwFBsqQS"
train_data = DocumentArray.pull('tll-train-data', show_progress=True)
query_data = DocumentArray.pull('tll-test-query-data', show_progress=True)
index_data = DocumentArray.pull('tll-test-index-data', show_progress=True)

train_data.summary()
```

<!-- #region id="mUoY1jq0klwk" -->
## Backbone model
Now let's see which backbone models we can use. You can see available models by calling `finetuner.describe_models()`.


For this example, we're gonna go with `resnet50`.
<!-- #endregion -->

<!-- #region id="xA7IIhIOk0h0" -->
## Fine-tuning

Now that we have the training and evaluation datasets loaded as `DocumentArray`s and selected our model, we can start our fine-tuning run.
<!-- #endregion -->

```python id="qGrHfz-2kVC7"
from finetuner.callback import EvaluationCallback

run = finetuner.fit(
    model='resnet50',
    train_data='tll-train-data',
    batch_size=128,
    epochs=5,
    learning_rate=1e-4,
    device='cuda',
    callbacks=[
        EvaluationCallback(
            query_data='tll-test-query-data',
            index_data='tll-test-index-data',
        )
    ],
)
```

<!-- #region id="9gvoWipMlG5P" -->
Let's understand what this piece of code does:

* As you can see, we have to provide the `model` which we picked before.
* We also set `run_name` and `description`, which are optional,
but recommended in order to retrieve your run easily and have some context about it.
* Furthermore, we had to provide names of the `train_data`.
* We set `TripletMarginLoss`.
* Additionally, we use `finetuner.callback.EvaluationCallback` for evaluation.
* Lastly, we set the number of `epochs` and provide a `learning_rate`.
<!-- #endregion -->

<!-- #region id="7ftSOH_olcak" -->
## Monitoring

Now that we've created a run, let's see its status. You can monitor the run by checking the status - `run.status()` - and the logs - `run.logs()` or `run.stream_logs()`. 
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
What you can do for now is to call `run.logs()` in the end of the run and see evaluation results:

```bash
  Training [5/5] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 76/76 0:00:00 0:03:15 • loss: 0.003
[16:39:13] DEBUG    Metric: 'model_average_precision' Value: 0.19598                                     __main__.py:202
           DEBUG    Metric: 'model_dcg_at_k' Value: 0.28571                                              __main__.py:202
           DEBUG    Metric: 'model_f1_score_at_k' Value: 0.04382                                         __main__.py:202
           DEBUG    Metric: 'model_hit_at_k' Value: 0.46013                                              __main__.py:202
           DEBUG    Metric: 'model_ndcg_at_k' Value: 0.28571                                             __main__.py:202
           DEBUG    Metric: 'model_precision_at_k' Value: 0.02301                                        __main__.py:202
           DEBUG    Metric: 'model_r_precision' Value: 0.19598                                           __main__.py:202
           DEBUG    Metric: 'model_recall_at_k' Value: 0.46013                                           __main__.py:202
           DEBUG    Metric: 'model_reciprocal_rank' Value: 0.19598                                       __main__.py:202
           INFO     Done ✨                                                                              __main__.py:204
           INFO     Saving fine-tuned models ...                                                         __main__.py:207
           INFO     Saving model 'model' in /usr/src/app/tuned-models/model ...                          __main__.py:218
           INFO     Pushing saved model to Jina AI Cloud ...                                             __main__.py:225
[16:39:41] INFO     Pushed model artifact ID: '62b33cb0037ad91ca7f20530'                                 __main__.py:231
           INFO     Finished 🚀                                                                          __main__.py:233                           __main__.py:248
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
And finally you can use the embeded `query` to find top-k visually related images within `index_data` as follows:
<!-- #endregion -->

```python id="tBYG9OKrpZ36"
query.match(index_data, limit=10, metric='cosine')
```
