# Text to image search using CLIP

This guide will showcase fine-tuning a `CLIP` model for text to image retrieval.

## Task overview
We'll be fine-tuning CLIP on a fashion captioning dataset which contains information about fashion products.

For each product the dataset contains a title and images of multiple variants of the product. We constructed a parent `Document` for each picture, which contains two chunks: an image document and a text document holding the description of the product.


## Preparing data
Training and evaluation data are already prepared and pushed to Hubble following the [instructions](../2_step_by_step/2_4_create_training_data.md).
You can either pull the data:
```python
from docarray import DocumentArray
train_data = DocumentArray.pull('clip-fashion-train-data')
eval_data = DocumentArray.pull('clip-fashion-eval-data')
```
Or specify given `DocumentArray` names (`clip-fashion-train-data` and `clip-fashion-eval-data`) directly to the finetuner.


## Login to Finetuner
As explained in the [Login to Jina ecosystem](../2_step_by_step/2_3_login_to_jina_ecosystem.md) section, first we need to login to Finetuner:
```python
import finetuner
finetuner.login()
```


## Choosing the model
Currently, we only support `openai/clip-vit-base-patch32` for text to image retrieval tasks. However, you can see all available models either in [the docs](../2_step_by_step/2_5_choose_back_bone.md) or by calling:
```python
finetuner.describe_models()
```

```bash
                                                                  Finetuner backbones                                                                   
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                            model ┃           task ┃ output_dim ┃ architecture ┃                                          description ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│                                         resnet50 │ image-to-image │       2048 │          CNN │                               Pretrained on ImageNet │
│                                        resnet152 │ image-to-image │       2048 │          CNN │                               Pretrained on ImageNet │
│                                  efficientnet_b0 │ image-to-image │       1280 │          CNN │                               Pretrained on ImageNet │
│                                  efficientnet_b4 │ image-to-image │       1280 │          CNN │                               Pretrained on ImageNet │
│                     openai/clip-vit-base-patch32 │  text-to-image │        768 │  transformer │ Pretrained on millions of text image pairs by OpenAI │
│                                  bert-base-cased │   text-to-text │        768 │  transformer │       Pretrained on BookCorpus and English Wikipedia │
│ sentence-transformers/msmarco-distilbert-base-v3 │   text-to-text │        768 │  transformer │           Pretrained on Bert, fine-tuned on MS Marco │
└──────────────────────────────────────────────────┴────────────────┴────────────┴──────────────┴──────────────────────────────────────────────────────┘
```


## Creating a fine-tuning job
Now that everything's ready, let's create a fine-tuning run!

```python
from finetuner.callback import BestModelCheckpoint, EvaluationCallback

run = finetuner.fit(
        model='openai/clip-vit-base-patch32',
        run_name='clip-fashion-1',
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
Let's understand what this piece of code does.
We start with providing `model`, `run_name`, names of training and evaluation data. We also provide some hyper-parameters
such as number of `epochs` and a `learning_rate`. Additionally, we use `BestModelCheckpoint` to save the best model after each epoch and `EvaluationCallback` for evaluation. Now let's move on to CLIP-specific arguments. We provided `image_modality`
and `text_modality` which are needed for `CLIP` model to distribute data across its two models properly. (More on this in the [create training data](../2_step_by_step/2_4_create_training_data.md) section).
We also need to provide a `CLIPloss` and set `multi_modal` to `True`.


Now that we've created a run, let's see its status.
```python
print(run.status())
```

```bash
{'status': 'CREATED', 'details': 'Run submitted and awaits execution'}
```


## Reconnect and retrieve the runs
Since some runs might take up to several hours/days, it's important to know how to reconnect to Finetuner and retrieve your run.

```python
import finetuner
finetuner.login()
run = finetuner.get_run('clip-fashion-1')
```

You can monitor the run by checking the status - `run.status()` or the logs - `run.logs()`. 

If your run has finished successfully, you can save fine-tuned models in the following way:
```python
run.save_model('clip-model')
```

## Evaluation
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
