---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Multilingual Text To Image search with MultilingualCLIP


Most text-image models are only able to provide embeddings for text in a single language, typically English. Multilingual CLIP models, however, are models that have been trained on multiple different languages. This allows the model the produce similar embeddings for the same sentence in multiple different languages.  

This guide will show you how to finetune a multilingual CLIP model for a text to image retrieval in non-English languages.

*Note, please consider switching to GPU/TPU Runtime for faster inference.*



## Install

```python
!pip install 'finetuner[full]'
```

## Task


We'll be finetuning multilingual CLIP on the `toloka-fashion` dataset, which contains information about fashion products, with all descriptions being in German.  

Each product in the dataset contains several attributes, we will be making use of the image and category attributes to create a [`Document`](https://docarray.jina.ai/fundamentals/document/#document) containing two [chunks](https://docarray.jina.ai/fundamentals/document/nested/#nested-structure), one containing the image and another containing the category of the product.


## Data
We will use the `toloka-fashion` dataset, which we have already pre-processed and made available on the Jina AI Cloud. You can access it by like so:

```python
import finetuner
from docarray import DocumentArray, Document

finetuner.login(force=True)
```

```python
train_data = DocumentArray.pull('toloka-fashion-train-data', show_progress=True)
eval_data = DocumentArray.pull('toloka-fashion-eval-data', show_progress=True)

train_data.summary()
```

## Backbone Model
Currently, we only support one multilingual CLIP model, which has been made available by [open-clip](https://github.com/mlfoundations/open_clip).


## Fine-tuning
Now that our data has been prepared, we can start our fine-tuning run.

```python
import finetuner

run = finetuner.fit(
    model='xlm-roberta-base-ViT-B-32::laion5b_s13b_b90k',
    train_data=train_data,
    eval_data=eval_data,
    epochs=5,
    learning_rate=1e-6,
    loss='CLIPLoss',
    device='cuda',
)
```

You may notice that this piece of code looks very similar to the one used to fine-tune regular clip models, as shown [here](https://finetuner.jina.ai/notebooks/text_to_image/). The only real difference is the data being provided and the model being used. 


## Monitoring

Now that we've created a run, let's see its status. You can monitor the run by checking the status - `run.status()` and - the logs - `run.logs()` or - `run.stream_logs()`. 

```python tags=[]
# note, the fine-tuning might takes 20~ minutes
for entry in run.stream_logs():
    print(entry)
```

<!-- #region -->
Since some runs might take up to several hours/days, it's important to know how to reconnect to Finetuner and retrieve your run.

```python
import finetuner

finetuner.login()
run = finetuner.get_run(run.name)
```

You can continue monitoring the run by checking the status - `finetuner.run.Run.status()` or the logs - `finetuner.run.Run.logs()`.
<!-- #endregion -->

<!-- #region -->
## Evaluating
Currently, we don't have a user-friendly way to get evaluation metrics from the {class}`~finetuner.callback.EvaluationCallback` we initialized previously.

```bash
           INFO     Done ✨                                                                              __main__.py:219
           INFO     Saving fine-tuned models ...                                                         __main__.py:222
           INFO     Saving model 'model' in /usr/src/app/tuned-models/model ...                          __main__.py:233
           INFO     Pushing saved model to Jina AI Cloud ...                                                    __main__.py:240
[10:38:14] INFO     Pushed model artifact ID: '62a1af491597c219f6a330fe'                                 __main__.py:246
           INFO     Finished 🚀                                                                          __main__.py:248
```

```{admonition} Evaluation of CLIP

In this example, we did not plug-in an `EvaluationCallback` since the callback can evaluate one model at one time.
In most cases, we want to evaluate two models: i.e. use `CLIPTextEncoder` to encode textual Documents as `query_data` while use `CLIPImageEncoder` to encode image Documents as `index_data`.
Then use the textual Documents to search image Documents.

We have done the evaulation for you in the table below.
```

|                   | Before Finetuning   | After Finetuning    |
|-------------------|---------------------|---------------------|
| average_precision | 0.449150592183874   | 0.5229004685258555  |
| dcg_at_k          | 0.6027663856128129  | 0.669843418638272   |
| f1_score_at_k     | 0.0796103896103896  | 0.08326118326118326 |
| hit_at_k          | 0.83                | 0.8683333333333333  |
| ndcg_at_k         | 0.5998242304751983  | 0.6652403194597005  |
| precision_at_k    | 0.04183333333333333 | 0.04375             |
| r_precision       | 0.4489283699616517  | 0.5226226907480778  |
| recall_at_k       | 0.8283333333333334  | 0.8666666666666667  |
| reciprocal_rank   | 0.44937281440609617 | 0.5231782463036333  |
<!-- #endregion -->

## Saving

After the run has finished successfully, you can download the tuned model on your local machine:

```python
artifact = run.save_artifact('m-clip-model')
```

## Inference

Now you saved the `artifact` into your host machine,
let's use the fine-tuned model to encode a new `Document`:

```python
text_da = DocumentArray([Document(text='setwas Text zum Codieren')])
image_da = DocumentArray([Document(uri='https://upload.wikimedia.org/wikipedia/commons/4/4e/Single_apple.png')])

clip_text_encoder = finetuner.get_model(artifact=artifact, select_model='clip-text')
clip_image_encoder = finetuner.get_model(artifact=artifact, select_model='clip-vision')

finetuner.encode(model=clip_text_encoder, data=text_da)
finetuner.encode(model=clip_image_encoder, data=image_da)

print(text_da.embeddings.shape)
print(image_da.embeddings.shape)
```

<!-- #region -->
```bash
(1, 512)
(1, 512)
```

```{admonition} what is select_model?
When fine-tuning CLIP, we are fine-tuning the CLIPVisionEncoder and CLIPTextEncoder in parallel.
The artifact contains two models: `clip-vision` and `clip-text`.
The parameter `select_model` tells finetuner which model to use for inference, in the above example,
we use `clip-text` to encode a Document with text content.
```

```{admonition} Inference with ONNX
In case you set `to_onnx=True` when calling `finetuner.fit` function,
please use `model = finetuner.get_model(artifact, is_onnx=True)`
```
<!-- #endregion -->
