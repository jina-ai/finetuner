(text-to-image)=
# Text-to-Image Search via CLIP

This guide will showcase fine-tuning a `CLIP` model for text to image retrieval.

## Task
We'll be fine-tuning CLIP on the [fashion captioning dataset](https://github.com/xuewyang/Fashion_Captioning) which contains information about fashion products.

For each product the dataset contains a title and images of multiple variants of the product. We constructed a parent [`Document`](https://docarray.jina.ai/fundamentals/document/#document) for each picture, which contains two [chunks](https://docarray.jina.ai/fundamentals/document/nested/#nested-structure): an image document and a text document holding the description of the product.


## Data
Our journey starts locally. We have to {ref}`prepare the data and push it to the cloud <create-training-data>` and Finetuner will be able to get the dataset by its name. For this example,
we already prepared the data, and we'll provide the names of training and evaluation data (`clip-fashion-train-data` and `clip-fashion-eval-data`) directly to Finetuner.

```{admonition} 
:class: tip
We don't require you to push data to the cloud by yourself. Instead of a name, you can provide a `DocumentArray` and Finetuner will do the job for you.
```


## Backbone model
Currently, we only support `openai/clip-vit-base-patch32` for text to image retrieval tasks. However, you can see all available models either in {ref}`choose backbone <choose-backbone>` section or by calling {meth}`~finetuner.describe_models()`.


## Fine-tuning
From now on, all the action happens in the cloud! 

First you need to {ref}`login to Jina ecosystem <login-to-jina-ecosystem>`:
```python
import finetuner
finetuner.login()
```

Now that everything's ready, let's create a fine-tuning run!

```python
import finetuner

run = finetuner.fit(
    model='openai/clip-vit-base-patch32',
    run_name='clip-fashion',
    train_data='clip-fashion-train-data',
    eval_data='clip-fashion-eval-data',
    epochs=5,
    learning_rate= 1e-5,
    loss='CLIPLoss',
    cpu=False,
)
```
Let's understand what this piece of code does:
```{admonition} finetuner.fit parameters
:class: tip
The only required arguments are `model` and `train_data`. We provide default values for others. Here is the [full list of the parameters](../../api/finetuner/#finetuner.fit). 
```
* We start with providing `model`, `run_name`, names of training and evaluation data.
* We also provide some hyper-parameters such as number of `epochs` and a `learning_rate`.
* Additionally, we use {class}`~finetuner.callback.BestModelCheckpoint` to save the best model after each epoch and {class}`~finetuner.callback.EvaluationCallback` for evaluation.

  
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

You can continue monitoring the run by checking the status - {meth}`~finetuner.run.Run.status()` or the logs - {meth}`~finetuner.run.Run.logs()`.


## Evaluating
Currently, we don't have a user-friendly way to get evaluation metrics from the {class}`~finetuner.callback.EvaluationCallback` we initialized previously.
What you can do for now is to call {meth}`~finetuner.run.Run.logs()` in the end of the run and see evaluation results:

```bash
           INFO     Done ✨                                                                              __main__.py:219
           INFO     Saving fine-tuned models ...                                                         __main__.py:222
           INFO     Saving model 'model' in /usr/src/app/tuned-models/model ...                          __main__.py:233
           INFO     Pushing saved model to Hubble ...                                                    __main__.py:240
[10:38:14] INFO     Pushed model artifact ID: '62a1af491597c219f6a330fe'                                 __main__.py:246
           INFO     Finished 🚀                                                                          __main__.py:248
```

```{admonition} Evaluation of CLIP
:class: hint

In this example, we did not plug-in an `EvaluationCallback` since the callback can evaluate one model at one time.
In most cases, we want to evaluate two models: i.e. use `CLIPTextEncoder` to encode textual Documents as `query_data` while use `CLIPImageEncoder` to encode image Documents as `index_data`.
Then use the textual Documents to search image Documents.

We have done the evaulation for you in the table below.
```

|                   | Before Finetuning | After Finetuning |
|:------------------|----------:|---------:|
| average_precision | 0.253423  | 0.415924 |
| dcg_at_k          | 0.902417  | 2.14489  |
| f1_score_at_k     | 0.0831918 | 0.241773 |
| hit_at_k          | 0.611976  | 0.856287 |
| ndcg_at_k         | 0.350172  | 0.539948 |
| precision_at_k    | 0.0994012 | 0.256587 |
| r_precision       | 0.231756  | 0.35847  |
| recall_at_k       | 0.108982  | 0.346108 |
| reciprocal_rank   | 0.288791  | 0.487505 |

## Saving

After the run has finished successfully, you can download the tuned model on your local machine:
```python
artifact = run.save_artifact('clip-model')
```

## Inference

Now you saved the `artifact` into your host machine,
let's use the fine-tuned model to encode a new `Document`:

```python
import finetuner
from docarray import Document, DocumentArray
# Load model from artifact
model = finetuner.get_model(artifact=artifact, select_model='clip-text')
# Prepare some text to encode
test_da = DocumentArray([Document(text='some text to encode')])
# Encoding will happen in-place in your `DocumentArray`
finetuner.encode(model=model, data=test_da)
print(test_da.embeddings)
```

```{admonition} what is select_model?
:class: hint
When fine-tuning CLIP, we are fine-tuning the CLIPVisionEncoder and CLIPTextEncoder in parallel.
The artifact contains two models: `clip-vision` and `clip-text`.
The parameter `select_model` tells finetuner which model to use for inference, in the above example,
we use `clip-text` to encode a Document with text content.
```

Check out [clip-as-service](https://clip-as-service.jina.ai/user-guides/finetuner/?highlight=finetuner#fine-tune-models) to learn how to plug-in a finetuned CLIP model to our CLIP specific service.

That's it! If you want to integrate the fine-tuned model into your Jina Flow, please check out {ref}`integrated with the Jina ecosystem <integrate-with-jina>`.