(image-to-image)=
# ResNet for image to image search

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
Out journey starts locally. We have to {ref}`prepare the data and push it to the cloud <create-training-data>` and Finetuner will be able to get the data by its name. For this example,
we already prepared the data, and we'll provide the names of training and evaluation data (`resnet-ttl-train-data` and `resnet-ttl-eval-data`) directly to Finetuner.

```{admonition} 
:class: tip
We don't require to push data to the cloud by yourself. Instead of a name, you can provide a `DocumentArray` and Finetuner will do the job for you.
```


## Backbone model
Now let's see which backbone models we can use. You can see available models either in {ref}`choose backbone <choose-backbone>` section or by calling [`finetuner.describe_models()`](../../api/finetuner/#finetuner.describe_models).


For this example, we're gonna go with `resnet50`.

## Fine-tuning
From now on, all the action happens in the cloud! 

First you need to {ref}`login to Jina ecosystem <login-to-jina-ecosystem>`:
```python
import finetuner
finetuner.login()
```

Now, you can easily start a fine-tuning job with [`finetuner.fit`]((../../api/finetuner/#finetuner.fit)):
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
* Additionally, we use `BestModelCheckpoint` to save the best model after each epoch and `EvaluationCallback` for evaluation.
* Lastly, we set number of `epochs` and provide a `learning_rate`.


## Monitoring

Let's check the status of the run.
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

You can continue monitoring the runs by checking the status - [`run.status()`](../../api/finetuner.run/#finetuner.run.Run.status) or the logs - [`run.logs()`](../../api/finetuner.run/#finetuner.run.Run.logs). 

## Evaluating
Currently, we don't have a user-friendly way to get evaluation metrics from the `EvaluationCallback` we initialized previously.
What you can do for now is to call [`run.logs()`](../../api/finetuner.run/#finetuner.run.Run.logs) in the end of the run and see evaluation results:

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

## Saving

After the run has finished successfully, you can download the tuned model on your local machine:
```python
run.save_model('resnet-model')
```
