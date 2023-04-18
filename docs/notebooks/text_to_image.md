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

<!-- #region id="3UCyCMPcvLGw" -->
# Text-to-Image Search via CLIP

<a href="https://colab.research.google.com/drive/1yKnmy2Qotrh3OhgwWRsMWPFwOSAecBxg?usp=sharing"><img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg"></a>

Traditionally, searching images from text (text-image-retrieval) relies heavily on human annotations, this is commonly referred to as *Text/Tag-based Image Retrieval (TBIR)*.

The [OpenAI CLIP](https://github.com/openai/CLIP) model maps the dense vectors extracted from text and image into the same semantic space and produces a strong zero-shot model to measure the similarity between text and images.

This guide will showcase fine-tuning a `CLIP` model for text-to-image retrieval.

*Note, please consider switching to GPU/TPU Runtime for faster inference.*

## Install
<!-- #endregion -->

```python id="vglobi-vvqCd"
!pip install 'finetuner[full]'
```

<!-- #region id="GXddluSIwCGW" -->
## Task
We'll be fine-tuning CLIP on the [fashion captioning dataset](https://github.com/xuewyang/Fashion_Captioning) which contains information about fashion products.

For each product, the dataset contains a title and images of multiple variants of the product. We constructed a parent [`Document`](https://finetuner.jina.ai/walkthrough/create-training-data/#preparing-a-documentarray) for each picture, which contains two chunks: an image document and a text document holding the description of the product.
<!-- #endregion -->

<!-- #region id="EVBez7dHwIye" -->
## Data
Our journey starts locally. We have to [prepare the data and push it to the Jina AI Cloud](https://finetuner.jina.ai/walkthrough/create-training-data/) and Finetuner will be able to get the dataset by its name. For this example,
we already prepared the data, and we'll provide the names of training and evaluation data (`fashion-train-data-clip` and `fashion-eval-data-clip`) directly to Finetuner.
In addition, we also provide labeled queries and an index of labeled documents for evaluating the retrieval capabilities of the resulting fine-tuned model stored in the datasets `fashion-eval-data-queries` and `fashion-eval-data-index`.


```{admonition} Push data to the cloud
We don't require you to push data to the Jina AI Cloud by yourself. Instead of a name, you can provide a `DocumentArray` and Finetuner will do the job for you.
When working with documents where images are stored locally, please call `doc.load_uri_to_blob()` to reduce network transmission and speed up training.
```
<!-- #endregion -->

```python id="vfPZBQVxxEHm"
import finetuner
from finetuner import DocumentArray, Document

finetuner.login(force=True)
```

```python id="cpIj7viExFti"
train_data = DocumentArray.pull('finetuner/fashion-train-data-clip', show_progress=True)
eval_data = DocumentArray.pull('finetuner/fashion-eval-data-clip', show_progress=True)
query_data = DocumentArray.pull('finetuner/fashion-eval-data-queries', show_progress=True)
index_data = DocumentArray.pull('finetuner/fashion-eval-data-index', show_progress=True)

train_data.summary()
```

<!-- #region id="AE87a5Nvwd7q" -->
## Backbone model
Currently, we support several CLIP variations from [open-clip](https://github.com/mlfoundations/open_clip) for text to image retrieval tasks.

However, you can see all available models either in [choose backbone](https://finetuner.jina.ai/walkthrough/choose-backbone/) section or by calling `finetuner.describe_models()`.
<!-- #endregion -->

<!-- #region id="81fh900Bxgkn" -->
## Fine-tuning

Now that we have the training and evaluation datasets loaded as `DocumentArray`s and selected our model, we can start our fine-tuning run.
<!-- #endregion -->

```python id="UDcpfybOv1dh"
from finetuner.callback import EvaluationCallback

run = finetuner.fit(
    model='clip-base-en',
    train_data='finetuner/fashion-train-data-clip',
    eval_data='finetuner/fashion-eval-data-clip',
    epochs=5,
    learning_rate= 1e-7,
    loss='CLIPLoss',
    device='cuda',
    callbacks=[
        EvaluationCallback(
            model='clip-text',
            index_model='clip-vision',
            query_data='finetuner/fashion-eval-data-queries',
            index_data='finetuner/fashion-eval-data-index',
        )
    ],
)
```

<!-- #region id="QPDmFdubxzUE" -->
Let's understand what this piece of code does:

* We start with providing `model`, names of training and evaluation data.
* We also provide some hyperparameters such as number of `epochs` and a `learning_rate`.
* We use `CLIPLoss` to optimize the CLIP model.
* We use an evaluation callback, which uses the `'clip-text'` model for encoding the text queries and the `'clip-vision'` model for encoding the images in `'fashion-eval-data-index'`.

<!-- #endregion -->

<!-- #region id="qKv3VcMKyG8d" -->
## Monitoring

Now that we've created a run, let's see its status. You can monitor the run by checking the status - `run.status()` and - the logs - `run.logs()` or - `run.stream_logs()`. 
<!-- #endregion -->

```python id="JX45y-2fxs4L"
# note, the fine-tuning might takes 20~ minutes
for entry in run.stream_logs():
    print(entry)
```

<!-- #region id="xi49YlQsyXbi" -->
Since some runs might take up to several hours/days, it's important to know how to reconnect to Finetuner and retrieve your run.

```python
import finetuner

finetuner.login()
run = finetuner.get_run(run.name)
```

You can continue monitoring the run by checking the status - `finetuner.run.Run.status()` or the logs - `finetuner.run.Run.logs()`.
<!-- #endregion -->

<!-- #region id="Xeq_aVRxyqlW" -->
## Evaluating

Our `EvaluationCallback` during fine-tuning ensures that after each epoch, an evaluation of our model is run. We can access the results of the last evaluation in the logs as follows `print(run.logs())`:

```bash
  Training [5/5] ━━━━ 195/195 0:00… 0:0… • loss: 2.419 • val_loss: 2.803
[13:32:41] INFO     Done ✨                              __main__.py:195
           DEBUG    Finetuning took 0 days, 0 hours 5 minutes and 30 seconds
           DEBUG    Metric: 'clip-text-to-clip-vision_precision_at_k' Value: 0.28532                                                   
           DEBUG    Metric: 'clip-text-to-clip-vision_hit_at_k' Value: 0.94282                                            
           DEBUG    Metric: 'clip-text-to-clip-vision_average_precision' Value: 0.53372                             
           DEBUG    Metric: 'clip-text-to-clip-vision_reciprocal_rank' Value: 0.67706                               
           DEBUG    Metric: 'clip-text-to-clip-vision_dcg_at_k' Value: 2.71247                                      
...
```

<!-- #endregion -->

<!-- #region id="h3qC3yAcy-Es" -->
## Saving

After the run has finished successfully, you can download the tuned model on your local machine:
<!-- #endregion -->

```python id="sucF7touyKo0"
artifact = run.save_artifact('clip-model')
```

<!-- #region id="8_VGjKq3zDx9" -->
## Inference

Now you saved the `artifact` into your host machine,
let's use the fine-tuned model to encode a new `Document`:
<!-- #endregion -->

```python id="v95QsuEyzE-B"
text_da = DocumentArray([Document(text='some text to encode')])
image_da = DocumentArray([Document(uri='https://upload.wikimedia.org/wikipedia/commons/4/4e/Single_apple.png')])

clip_text_encoder = finetuner.get_model(artifact=artifact, select_model='clip-text')
clip_image_encoder = finetuner.get_model(artifact=artifact, select_model='clip-vision')

finetuner.encode(model=clip_text_encoder, data=text_da)
finetuner.encode(model=clip_image_encoder, data=image_da)

print(text_da.embeddings.shape)
print(image_da.embeddings.shape)
```

<!-- #region id="LzMbR7VgzXtA" -->
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

<!-- #region id="LHyMm_M1zxdt" -->
## Advanced: WiSE-FT 

WiSE-FT, proposed by Mitchell et al. in [Robust fine-tuning of zero-shot models](https://arxiv.org/abs/2109.01903),
has been proven to be an effective way for fine-tuning models with a strong zero-shot capability,
such as CLIP.
As was introduced in the paper:

> Large pre-trained models such as CLIP or ALIGN offer consistent accuracy across a range of data distributions when performing zero-shot inference (i.e., without fine-tuning on a specific dataset). Although existing fine-tuning methods substantially improve accuracy on a given target distribution, they often reduce robustness to distribution shifts. We address this tension by introducing a simple and effective method for improving robustness while fine-tuning: ensembling the weights of the zero-shot and fine-tuned models (WiSE-FT).

Finetuner allows you to apply WiSE-FT easily,
all you need to do is use the `WiSEFTCallback`.
Finetuner will trigger the callback when the fine-tuning job is finished and merge the weights between the pre-trained model and the fine-tuned model:

```diff
from finetuner.callback import WiSEFTCallback

run = finetuner.fit(
    model='clip-base-en',
    ...,
    loss='CLIPLoss',
-   callbacks=[],
+   callbacks=[WiSEFTCallback(alpha=0.5)],
)
```

The value you set for `alpha` should be greater than or equal to 0 and less than or equal to 1:

+ if `alpha` is a float between 0 and 1, we merge the weights between the pre-trained model and the fine-tuned model.
+ if `alpha` is 0, the fine-tuned model is identical to the pre-trained model.
+ if `alpha` is 1, the pre-trained weights will not be utilized.


That's it! Check out [clip-as-service](https://clip-as-service.jina.ai/user-guides/finetuner/?highlight=finetuner#fine-tune-models) to learn how to plug in a fine-tuned CLIP model to our CLIP-specific service.
<!-- #endregion -->

<!-- #region id="tpm8eVRFX20B" -->
## Before and after
We can directly compare the results of our fine-tuned model with a pre-trained clip model by displaying the matches each model has for the same query. While the differences between the results of the two models are quite subtle for some queries, the examples below clearly show that finetuning increases the quality of the search results:
<!-- #endregion -->

<!-- #region id="C30UVpHDX4HF" -->

```plaintext
Results for query: "nightingale tee jacket" using a zero-shot model (top) and the fine-tuned model (bottom)
```
![clip-example-pt](https://user-images.githubusercontent.com/6599259/212634395-6f336d39-cda7-425d-80a2-10facae3b824.png)

![clip-example-ft](https://user-images.githubusercontent.com/6599259/212634112-a44c6c4c-2cc1-4dfb-8e29-0d02b2d6b95c.png)

<!-- #endregion -->
