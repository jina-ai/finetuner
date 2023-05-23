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

<!-- #region id="72867ba9-6a8c-4b14-acbf-487ea0a61836" -->
# Multilingual Text-to-Image Search with MultilingualCLIP

<a href="https://colab.research.google.com/drive/10Wldbu0Zugj7NmQyZwZzuorZ6SSAhtIo"><img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg"></a>

<!-- #endregion -->

<!-- #region id="f576573b-a48f-4790-817d-e99f8bd28fd0" -->
Most text-image models are only able to provide embeddings for text in a single language, typically English. Multilingual CLIP models, however, are models that have been trained on multiple different languages. This allows the model to produce similar embeddings for the same sentence in multiple different languages.  

This guide will show you how to finetune a multilingual CLIP model for a text to image retrieval task in non-English languages.

*Note, Check the runtime menu to be sure you are using a GPU/TPU instance, or this code will run very slowly.*

<!-- #endregion -->

<!-- #region id="ed1e7d55-a458-4dfd-8f4c-eeb02521c221" -->
## Install
<!-- #endregion -->

```python id="9261d0a7-ad6d-461f-bdf7-54e9804cc45d"
!pip install 'finetuner[full]'
```

<!-- #region id="11f13ad8-e0a7-4ba6-b52b-f85dd221db0f" -->
## Task
<!-- #endregion -->

<!-- #region id="ed1f88d4-f140-48d4-9d20-00e628c73e38" -->
We'll be fine-tuning multilingual CLIP on the electronics section of the [German Fashion12k dataset](https://github.com/Toloka/Fashion12K_german_queries), which contains images and descriptions of fashion products in German.

The images are a subset of the [xthan/fashion-200k dataset](https://github.com/xthan/fashion-200k), and we have commissioned their human annotations via crowdsourcing platform. Annotations were made in two steps.  First, we passed the 12,000 images to annotators in their large international user community, who added descriptive captions.

Each product in the dataset contains several attributes, we will be making use of the image and captions to create a [`Document`](https://finetuner.jina.ai/walkthrough/create-training-data/#preparing-a-documentarray) containing two chunks, one containing the image and another containing the category of the product.
<!-- #endregion -->

<!-- #region id="2a40f0b1-7272-4ae6-9d0a-f5c8d6d534d8" -->
## Data
We will use the `DE-Fashion-Image-Text-Multimodal-train` dataset, which we have already pre-processed and made available on the Jina AI Cloud. You can access it using `DocArray.pull`:
<!-- #endregion -->

```python id="4420a4ac-531a-4db3-af75-ebb58d8f828b"
import finetuner
from finetuner import DocumentArray, Document

finetuner.login(force=True)
```

```python id="bab5c3fb-ee75-4818-bd18-23c7a5983e1b"
train_data = DocumentArray.pull('finetuner/DE-Fashion-Image-Text-Multimodal-train', show_progress=True)
eval_data = DocumentArray.pull('finetuner/DE-Fashion-Image-Text-Multimodal-test', show_progress=True)

query_data = DocumentArray.pull('finetuner/DE-Fashion-Image-Text-Multimodal-query', show_progress=True)
index_data = DocumentArray.pull('finetuner/DE-Fashion-Image-Text-Multimodal-index', show_progress=True)

train_data.summary()
```

<!-- #region id="3b859e9c-99e0-484b-98d5-643ad51de8f0" -->
## Backbone Model
Currently, we only support one multilingual CLIP model. This model is the `xlm-roberta-base-ViT-B-32` from [open-clip](https://github.com/mlfoundations/open_clip), which has been trained on the [`laion5b` dataset](https://github.com/LAION-AI/laion5B-paper).
<!-- #endregion -->

<!-- #region id="0b57559c-aa55-40ff-9d05-f061dfb01354" -->
## Fine-tuning
Now that our data has been prepared, we can start our fine-tuning run.
<!-- #endregion -->

```python id="a0cba20d-e335-43e0-8936-d926568034b3"
from finetuner.callback import EvaluationCallback, WandBLogger

run = finetuner.fit(
    model='clip-base-multi',
    train_data='finetuner/DE-Fashion-Image-Text-Multimodal-train',
    epochs=5,
    learning_rate=1e-6,
    loss='CLIPLoss',
    device='cuda',
    callbacks=[
        EvaluationCallback(
            query_data='finetuner/DE-Fashion-Image-Text-Multimodal-query',
            index_data='finetuner/DE-Fashion-Image-Text-Multimodal-index',
            model='clip-text',
            index_model='clip-vision'
        ),
        WandBLogger(),
    ]
)
```

<!-- #region id="6be36da7-452b-4450-a5d5-6cae84522bb5" -->
Let's understand what this piece of code does:

* We start with providing `model`, names of training and evaluation data.
* We also provide some hyper-parameters such as number of `epochs` and a `learning_rate`.
* We use `CLIPLoss` to optimize the CLIP model.
* We use `finetuner.callback.EvaluationCallback` for evaluation.
* We then use the `finetuner.callback.WandBLogger` to display our results.
<!-- #endregion -->

<!-- #region id="923e4206-ac60-4a75-bb3d-4acfc4218cea" -->
## Monitoring

Now that we've created a run, let's see its status. You can monitor the run by checking the status - `run.status()` - and the logs - `run.logs()` or `run.stream_logs()`. 
<!-- #endregion -->

```python id="56d020bf-8095-4a83-a532-9b6c296e985a" tags=[]
# note, the fine-tuning might takes 20~ minutes
for entry in run.stream_logs():
    print(entry)
```

<!-- #region id="b58930f1-d9f5-43d3-b852-5cbaa04cb1aa" -->
Since some runs might take up to several hours/days, it's important to know how to reconnect to Finetuner and retrieve your run.

```python
import finetuner

finetuner.login()
run = finetuner.get_run(run.name)
```

You can continue monitoring the run by checking the status - `finetuner.run.Run.status()` or the logs `finetuner.run.Run.logs()`.
<!-- #endregion -->

<!-- #region id="f0b81ec1-2e02-472f-b2f4-27085bb041cc" -->
## Evaluating
Once the run is finished, the metrics are calculated by the {class}`~finetuner.callback.EvaluationCallback` and plotted using the {class}`~finetuner.callback.WandBLogger` callback. These plots can be accessed using the link provided in the logs once finetuning starts:

```bash
           INFO     Finetuning ... 
wandb: Currently logged in as: anony-mouse-448424. Use `wandb login --relogin` to force relogin
wandb: Tracking run with wandb version 0.13.5
wandb: Run data is saved locally in <path-to-file>
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run ancient-galaxy-2
wandb:  View project at <link-to-project>
wandb:  View run at <link-to-run>
[07:48:21] INFO     Done ✨                                                                              __main__.py:195
           DEBUG    Finetuning took 0 days, 0 hours 8 minutes and 19 seconds                             __main__.py:197
           DEBUG    Metric: 'clip-text-to-clip-vision_precision_at_k' Value: 0.04035                     __main__.py:206
           DEBUG    Metric: 'clip-text-to-clip-vision_hit_at_k' Value: 0.79200                           __main__.py:206
           DEBUG    Metric: 'clip-text-to-clip-vision_average_precision' Value: 0.41681                  __main__.py:206
           DEBUG    Metric: 'clip-text-to-clip-vision_reciprocal_rank' Value: 0.41773                    __main__.py:206
           DEBUG    Metric: 'clip-text-to-clip-vision_dcg_at_k' Value: 0.57113                           __main__.py:206
           INFO     Building the artifact ...                                                            __main__.py:208
           INFO     Pushing artifact to Jina AI Cloud ...                                                __main__.py:234
[08:02:33] INFO     Artifact pushed under ID '63b52b5b3278416c15353bf3'                                  __main__.py:236
           DEBUG    Artifact size is 2599.190 MB                                                         __main__.py:238
           INFO     Finished 🚀                                                                          __main__.py:239
```

The generated plots should look like this:

![WandB-mclip](https://user-images.githubusercontent.com/6599259/212645881-20071aba-8643-4878-bc53-97eb6f766bf0.png)

<!-- #endregion -->

<!-- #region id="2b8da34d-4c14-424a-bae5-6770f40a0721" -->
## Saving

After the run has finished successfully, you can download the tuned model on your local machine:
<!-- #endregion -->

```python id="0476c03f-838a-4589-835c-60d1b7f3f893"
artifact = run.save_artifact('mclip-model')
```

<!-- #region id="baabd6be-8660-47cc-a48d-feb43d0a507b" -->
## Inference

Now you saved the `artifact` into your host machine,
let's use the fine-tuned model to encode a new `Document`:
<!-- #endregion -->

```python id="fe43402f-4191-4343-905c-c75c64694662"
text_da = DocumentArray([Document(text='setwas Text zum Codieren')])
image_da = DocumentArray([Document(uri='https://upload.wikimedia.org/wikipedia/commons/4/4e/Single_apple.png')])

mclip_text_encoder = finetuner.get_model(artifact=artifact, select_model='clip-text')
mclip_image_encoder = finetuner.get_model(artifact=artifact, select_model='clip-vision')

finetuner.encode(model=mclip_text_encoder, data=text_da)
finetuner.encode(model=mclip_image_encoder, data=image_da)

print(text_da.embeddings.shape)
print(image_da.embeddings.shape)
```

<!-- #region id="ff2e7818-bf11-4179-a34d-d7b790b0db12" -->
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

<!-- #region id="38bc9069-0f0e-47c6-8560-bf77ad200774" -->
## Before and after
We can directly compare the results of our fine-tuned model with an untrained multilingual clip model by displaying the matches each model has for the same query, while the differences between the results of the two models are quite subtle for some queries, the examples below clearly show that fine-tuning increases the quality of the search results:
<!-- #endregion -->

<!-- #region id="e69fdfb2-6482-45fb-9c4d-41e548ef8f06" -->
```plaintext
results for query: "Spitzen-Midirock Teilfutter Schwarz" (Lace midi skirt partial lining black) using a zero-shot model and the fine-tuned model
```

before             |  after
:-------------------------:|:-------------------------:
![mclip-example-pt-1](https://jina-ai-gmbh.ghost.io/content/images/2022/12/mclip-before.png)  |  ![mclip-example-ft-1](https://jina-ai-gmbh.ghost.io/content/images/2022/12/mclip-after.png)



<!-- #endregion -->
