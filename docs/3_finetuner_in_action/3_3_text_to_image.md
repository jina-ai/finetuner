(text-to-image)=
# CLIP for text to image search

This guide will showcase fine-tuning a `CLIP` model for text to image retrieval.

## Task
We'll be fine-tuning CLIP on a fashion captioning dataset which contains information about fashion products.

For each product the dataset contains a title and images of multiple variants of the product. We constructed a parent `Document` for each picture, which contains two chunks: an image document and a text document holding the description of the product.


## Data
Out journey starts locally. We have to {ref}`prepare the data and push it to the cloud <create-training-data>` and Finetuner will be able to get the data by its name. For this example,
we already prepared the data, and we'll provide the names of training and evaluation data (`clip-fashion-train-data` and `clip-fashion-eval-data`) directly to Finetuner.

```{admonition} 
:class: tip
We don't require to push data to the cloud by yourself. Instead of a name, you can provide a `DocumentArray` and Finetuner will do the job for you.
```


## Backbone model
Currently, we only support `openai/clip-vit-base-patch32` for text to image retrieval tasks. However, you can see all available models either in {ref}`choose backbone <choose-backbone>` section or by calling [`finetuner.describe_models()`](../../api/finetuner/#finetuner.describe_models).


## Fine-tuning
From now on, all the action happens in the cloud! 

First you need to {ref}`login to Jina ecosystem <login-to-jina-ecosystem>`:
```python
import finetuner
finetuner.login()
```

Now that everything's ready, let's create a fine-tuning run!

```python
from finetuner.callback import BestModelCheckpoint, EvaluationCallback

run = finetuner.fit(
        model='openai/clip-vit-base-patch32',
        run_name='clip-fashion',
        train_data='clip-fashion-train-data',
        eval_data='clip-fashion-eval-data',
        epochs=5,
        learning_rate= 1e-5,
        callbacks=[BestModelCheckpoint(), EvaluationCallback(query_data='clip-fashion-eval-data')],
        loss='CLIPLoss',
        image_modality='image',
        text_modality='text',
        multi_modal=True,
    )
```
Let's understand what this piece of code does:
```{admonition} finetuner.fit parameters
:class: tip
The only required arguments are `model` and `train_data`. We provide default values for others. Here is the [full list of the parameters](../../api/finetuner/#finetuner.fit). 
```
* We start with providing `model`, `run_name`, names of training and evaluation data.
* We also provide some hyper-parameters such as number of `epochs` and a `learning_rate`.
* Additionally, we use `BestModelCheckpoint` to save the best model after each epoch and `EvaluationCallback` for evaluation.
* Now let's move on to CLIP-specific arguments: We provided `image_modality`
and `text_modality` which are needed for `CLIP` model to [distribute data across its two models properly](../2_step_by_step/2_4_create_training_data.md).
We also need to provide a `CLIPloss` and set `multi_modal` to `True`.

  
## Monitoring

We created a run! Now let's see its status.
```python
print(run.status())
```

```bash
{'status': 'CREATED', 'details': 'Run submitted and awaits execution'}
```

Since some runs might take up to several hours/days, it's important to know how to reconnect to Finetuner and retrieve your run.

```python
import finetuner
finetuner.login()
run = finetuner.get_run('clip-fashion')
```

You can continue monitoring the run by checking the status - [`run.status()`](../../api/finetuner.run/#finetuner.run.Run.status) or the logs - [`run.logs()`](../../api/finetuner.run/#finetuner.run.Run.logs). 


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
run.save_model('clip-model')
```
