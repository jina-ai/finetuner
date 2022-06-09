# Image to image search using ResNet

This guide will demonstrate how to fine-tune a ResNet model for image to image retrieval.

## Task Overview
More specifically, we will fine-tune ResNet50 on [Totally Looks Like Dataset](https://sites.google.com/view/totally-looks-like-dataset).
The dataset consists of 6016 pairs of images (12032 in total).


```{admonition} About TTL
:class: tip
Totally-Looks-Like is a dataset and benchmark challenging machine-learned representations to reproduce human perception of image similarity. As shown below, each image patch in the left has a corresponding similar image patch in the right. 
```

The dataset consists of pairs of images, these are the positive pairs. Negative pairs are constructed by taking two different images, i.e. images that are not in the same pair initially. Following this approach, we construct triplets and use the `TripletLoss`.
After fine-tuning, the embeddings of positive pairs are expected to be pulled closer, while the embeddings for negative pairs are expected to be pushed away.


## Preparing data
Training and evaluation data are already prepared and pushed to Hubble following the [instructions](../2_step_by_step/2_4_create_training_data.md).
You can either pull the data:
```python
from docarray import DocumentArray
train_data = DocumentArray.pull('resnet-ttl-train-data')
eval_data = DocumentArray.pull('resnet-ttl-eval-data')
```
Or specify given `DocumentArray` names (`resnet-ttl-train-data` and `resnet-ttl-eval-data`) directly to Finetuner.

## Choosing the model
Now let's see what backbone models we can use. You can see available models either in [the docs](../2_step_by_step/2_5_choose_back_bone.md) or by calling `finetuner.describe_models()`.


For this example, we're gonna go with `resnet50`.

## Creating a fine-tuning job
You can easily start a fine-tuning run with `finetuner.fit`.

```{admonition} Login to Jina Cloud
:class: tip
Before creating a run, you need to [Login to Jina ecosystem](../2_step_by_step/2_3_login_to_jina_ecosystem.md) by calling `finetuner.login()`.
```

```python
from finetuner.callback import BestModelCheckpoint, EvaluationCallback

run = finetuner.fit(
        model='resnet50',
        run_name='resnet-ttl',
        description='fine-tune the whole model.',
        train_data='resnet-ttl-train-data',
        eval_data='resnet-ttl-eval-data',
        loss='TripletMarginLoss',
        callbacks=[BestModelCheckpoint(), EvaluationCallback(query_data='resnet-ttl-eval-data')],
        epochs=6,
        learning_rate=0.001,
    )
```
Now, let's understand what this piece of code does. 
```{admonition} finetuner.fit parameters
:class: tip
The only required arguments are `model` and `train_data`. We provide default values for others. Here is the [full list of the parameters](../../api/finetuner/#finetuner.fit). 
```
As you can see, we have to provide the `model` which we picked before. We also set `run_name` and `description`, which are optional,
but recommended in order to retrieve your run easily and have some context about it. Furthermore, we had to provide `train_data` and `eval_data`. As you can see,
we used the names of the `DocumentArray`s that are already on Hubble, but we could also pass a `DocumentArray` object itself, which will be automatically uploaded to Hubble. As stated before, we want to use the `TripletLoss`, and that's what `loss='TripletMarginLoss'` corresponds to.
Additionally, we use `BestModelCheckpoint` to save the best model after each epoch and `EvaluationCallback` for evaluation. Lastly, we set number of `epochs` and provide a `learning_rate`.


## Monitoring your runs

Let's check the status of our runs.
```python
print(run.status())
```

```bash
{'status': 'CREATED', 'details': 'Run submitted and awaits execution'}
```

Since some runs might take up to several hours/days, it's important to know how to reconnect to Finetuner and retrieve your runs.

```python
import finetuner
finetuner.login()

run = finetuner.get_run('resnet-ttl')
```

You can continue monitoring the runs by checking the status - `run.status()` or the logs - `run.logs()`. 

## Saving your model

If your runs have finished successfully, you can save fine-tuned models in the following way:
```python
run.save_model('resnet-model')
```

## Evaluating your model
Currently, we don't have a user-friendly way to get evaluation metrics from the `EvaluationCallback` we initialized previously.
What you can do for now is to call `run.logs()` in the end of the run and see evaluation results:

```bash
[10:37:49] DEBUG    Metric: 'model_average_precision' Value: 0.30105                                     __main__.py:217
           DEBUG    Metric: 'model_dcg_at_k' Value: 0.43933                                              __main__.py:217
           DEBUG    Metric: 'model_f1_score_at_k' Value: 0.06273                                         __main__.py:217
           DEBUG    Metric: 'model_hit_at_k' Value: 0.69000                                              __main__.py:217
           DEBUG    Metric: 'model_ndcg_at_k' Value: 0.43933                                             __main__.py:217
           DEBUG    Metric: 'model_precision_at_k' Value: 0.03450                                        __main__.py:217
           DEBUG    Metric: 'model_r_precision' Value: 0.30105                                           __main__.py:217
           DEBUG    Metric: 'model_recall_at_k' Value: 0.34500                                           __main__.py:217
           DEBUG    Metric: 'model_reciprocal_rank' Value: 0.30105                                       __main__.py:217
           INFO     Done ✨                                                                              __main__.py:219
           INFO     Saving fine-tuned models ...                                                         __main__.py:222
           INFO     Saving model 'model' in /usr/src/app/tuned-models/model ...                          __main__.py:233
           INFO     Pushing saved model to Hubble ...                                                    __main__.py:240
[10:38:14] INFO     Pushed model artifact ID: '62a1af491597c219f6a330fe'                                 __main__.py:246
           INFO     Finished 🚀                                                                          __main__.py:248
```
