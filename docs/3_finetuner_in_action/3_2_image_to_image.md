# Image to image search using ResNet

This guide will demonstrate how to fine-tune a ResNet model for image to image retrieval.

## Task Overview
More specifically, we will fine-tune ResNet50 on [Totally Looks Like Dataset](https://sites.google.com/view/totally-looks-like-dataset).
The dataset consists of 6016 pairs of images (12032 in total).


```{admonition} About TTL
:class: tip
Totally-Looks-Like is a dataset and benchmark challenging machine-learned representations to reproduce human perception of image similarity. As shown below, each image patch in the left has a corresponding similar image patch in the right. 
```

<p align="center">
  <img src="https://finetuner.jina.ai/_static/ttl_overview.png" />
</p>

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

## Login to Finetuner
As explained in the [Login to Jina ecosystem](../2_step_by_step/2_3_login_to_jina_ecosystem.md) section, first we need to login to Finetuner:
```python
import finetuner
finetuner.login()
```

## Choosing the model
Now let's see what backbone models we can use. You can see available models either in [the docs](../2_step_by_step/2_5_choose_back_bone.md) or by calling:
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

For this example, we're gonna go with `resnet50`.

## Creating a fine-tuning job
You can easily start a fine-tuning run with `finetuner.fit`. With Finetuner you can create several fine-tuning jobs which will run in parallel.
Let's use this advantage now and create two runs with different `learning_rate` values.

```python
from finetuner.callbacks import BestModelCheckpoint

run1 = finetuner.fit(
        model='resnet50',
        run_name='resnet-ttl-1',
        description='fine-tune the whole model.',
        train_data='resnet-ttl-train-data',
        eval_data='resnet-ttl-eval-data',
        loss='TripletMarginLoss',
        callbacks=[BestModelCheckpoint()],
        epochs=6,
        learning_rate=0.001,
    )

run2 = finetuner.fit(
        model='resnet50',
        run_name='resnet-ttl-2',
        description='freeze all weights, and only fine-tune the additional MLP layer.',
        train_data='resnet-ttl-train-data',
        eval_data='resnet-ttl-eval-data',
        loss='TripletMarginLoss',
        callbacks=[BestModelCheckpoint()],
        epochs=6,
        learning_rate=0.01,
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
We used `BestModelCheckpoint` callback and set `epochs` to 6. Lastly, we have `learning_rate` which has a different value for each run.

Let's check the status of our runs.
```python
print(run.name, '-', run1.status())
print(run.name, '-', run2.status())
```

```bash
resnet-ttl-1 - {'status': 'CREATED', 'details': 'Run submitted and awaits execution'}
resnet-ttl-2 - {'status': 'CREATED', 'details': 'Run submitted and awaits execution'}
```

## Reconnect and retrieve the runs
Since some runs might take up to several hours/days, it's important to know how to reconnect to Finetuner and retrieve your runs.

```python
import finetuner
finetuner.login()

run1 = finetuner.get_run('resnet-ttl-1')
run2 = finetuner.get_run('resnet-ttl-2')
```

You can monitor the runs by checking the status - `run.status()` or the logs - `run.logs()`. 

If your runs have finished successfully, you can save fine-tuned models in the following way:
```python
run1.save_model('without-freezing')
run2.save_model('freezed-model')
```

## Evaluation
Currently, we don't have a user-friendly way to get evaluation metrics from the `EvaluationCallback` we initialized previously.
What you can do for now is to call `run.logs()` in the end of the run and see evaluation results in the logs.
