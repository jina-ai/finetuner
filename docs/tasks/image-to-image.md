(image-to-image)=
# Image-to-Image Search via ResNet50

This guide will demonstrate how to fine-tune a ResNet model for image to image retrieval.

## Task
More specifically, we will fine-tune ResNet50 on [Totally Looks Like Dataset](https://sites.google.com/view/totally-looks-like-dataset).
The dataset consists of 6016 pairs of images (12032 in total).


```{admonition} About TTL
:class: tip
Totally-Looks-Like is a dataset and benchmark challenging machine-learned representations to reproduce human perception of image similarity. As shown below, each image patch in the left has a corresponding similar image patch in the right. 
```

The dataset consists of pairs of images, these are the positive pairs. Negative pairs are constructed by taking two different images, i.e. images that are not in the same pair initially. Following this approach, we construct triplets and use the `TripletLoss`.
After fine-tuning, the embeddings of positive pairs are expected to be pulled closer, while the embeddings for negative pairs are expected to be pushed away.


## Data
Our journey starts locally. We have to {ref}`prepare the data and push it to the cloud <create-training-data>` and Finetuner will be able to get the dataset by its name. For this example,
we already prepared the data, and we'll provide the names of training and evaluation data (`resnet-ttl-train-data` and `resnet-ttl-eval-data`) directly to Finetuner.

```{admonition} 
:class: tip
We don't require you to push data to the cloud by yourself. Instead of a name, you can provide a `DocumentArray` and Finetuner will do the job for you.
```

```{important}
When working with Document where images are stored locally, please call `doc.load_uri_to_image_tensor(width=224, height=224)` or another image shape to reduce network transmission and speed up training.
```


## Backbone model
Now let's see which backbone models we can use. You can see available models either in {ref}`choose backbone <choose-backbone>` section or by calling {meth}`~finetuner.describe_models()`.


For this example, we're gonna go with `resnet50`.

## Fine-tuning
From now on, all the action happens in the cloud! 

First you need to {ref}`login to Jina ecosystem <login-to-jina-ecosystem>`:
```python
import finetuner
finetuner.login()
```

Now, you can easily start a fine-tuning job with {meth}`~finetuner.fit`:
```python
from finetuner.callback import EvaluationCallback

run = finetuner.fit(
    model='resnet50',
    run_name='resnet-tll',
    train_data='tll-train-da',
    batch_size=128,
    epochs=5,
    learning_rate=1e-5,
    cpu=False,
    callbacks=[
        EvaluationCallback(
            query_data='tll-test-query-da',
            index_data='tll-test-index-da',
        )
    ],
)
```
Let's understand what this piece of code does:
```{admonition} finetuner.fit parameters
:class: tip
The only required arguments are `model` and `train_data`. We provide default values for others. Here is the [full list of the parameters](../../api/finetuner/#finetuner.fit). 
```
* As you can see, we have to provide the `model` which we picked before.
* We also set `run_name` and `description`, which are optional,
but recommended in order to retrieve your run easily and have some context about it.
* Furthermore, we had to provide names of the `train_data` and `eval_data`.
* We set `TripletMarginLoss`.
* Additionally, we use {class}`~finetuner.callback.BestModelCheckpoint` to save the best model after each epoch and {class}`~finetuner.callback.EvaluationCallback` for evaluation.
* Lastly, we set number of `epochs` and provide a `learning_rate`.


## Monitoring

Let's check the status of the run.
```python
print(run.status())
```

```bash
{'status': 'CREATED', 'details': 'Run submitted and awaits execution'}
```

Since some runs might take up to several hours, it's important to know how to reconnect to Finetuner and retrieve your runs.

```python
import finetuner
finetuner.login()

run = finetuner.get_run('resnet-tll')
```

You can continue monitoring the runs by checking the status - {meth}`~finetuner.run.Run.status()` or the logs - {meth}`~finetuner.run.Run.logs()`. 

## Evaluating
Currently, we don't have a user-friendly way to get evaluation metrics from the {class}`~finetuner.callback.EvaluationCallback` we initialized previously.
What you can do for now is to call {meth}`~finetuner.run.Run.logs()` in the end of the run and see evaluation results:

```bash
  Training [5/5] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 76/76 0:00:00 0:03:15 • loss: 0.003
[16:39:13] DEBUG    Metric: 'model_average_precision' Value: 0.16603                                     __main__.py:202
           DEBUG    Metric: 'model_dcg_at_k' Value: 0.23632                                              __main__.py:202
           DEBUG    Metric: 'model_f1_score_at_k' Value: 0.03544                                         __main__.py:202
           DEBUG    Metric: 'model_hit_at_k' Value: 0.37209                                              __main__.py:202
           DEBUG    Metric: 'model_ndcg_at_k' Value: 0.23632                                             __main__.py:202
           DEBUG    Metric: 'model_precision_at_k' Value: 0.01860                                        __main__.py:202
           DEBUG    Metric: 'model_r_precision' Value: 0.16603                                           __main__.py:202
           DEBUG    Metric: 'model_recall_at_k' Value: 0.37209                                           __main__.py:202
           DEBUG    Metric: 'model_reciprocal_rank' Value: 0.16603                                       __main__.py:202
           INFO     Done ✨                                                                              __main__.py:204
           INFO     Saving fine-tuned models ...                                                         __main__.py:207
           INFO     Saving model 'model' in /usr/src/app/tuned-models/model ...                          __main__.py:218
           INFO     Pushing saved model to Hubble ...                                                    __main__.py:225
[16:39:41] INFO     Pushed model artifact ID: '62b33cb0037ad91ca7f20530'                                 __main__.py:231
           INFO     Finished 🚀                                                                          __main__.py:233                           __main__.py:248
```

## Saving

After the run has finished successfully, you can download the tuned model on your local machine:
```python
run.save_artifact('resnet-model')
```

## Inference

Now you saved the `artifact` into your host machine,
let's use fine-tuned model to encode a new `Document`.

```python
import finetuner
from docarray import Document, DocumentArray
# Load model from artifact
model = finetuner.get_model(artifact=artifact)
# Prepare some text to encode, change the placeholder image uri to an image on your machine
test_da = DocumentArray([Document(uri='my-image.png')])
# Encoding will happen in-place in your `DocumentArray`
finetuner.encode(model=model, data=test_da)
print(test_da.embeddings)
```

That's it! If you want to integrate fine-tuned model into your Jina Flow, please check out {ref}`integrated with the Jina ecosystem <integrate-with-jina>`.