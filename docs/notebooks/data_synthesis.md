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

<!-- #region id="RDjy9CrsuHH5" -->
# Data Synthesis

<a href="https://colab.research.google.com/drive/1sX5K0eophlHXu1S7joysZJUj1zfh28Gi?usp=sharing"><img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg"></a>

When using Finetuner, each item in your training data must either have a label, or have a similarity score comparing it to some other item. See the Finetuner documentation on [preparing training data](https://finetuner.jina.ai/walkthrough/create-training-data/).
If your data is not labelled, and you don't want to spend time manually organizing and labelling it, you can use the `finetuner.synthesize` function to automatically construct a dataset that can be used in training.

This guide will walk you through the process of using the `finetuner.synthesize` function, as well as how to use its output for training.

![synthesis_flowchart](https://user-images.githubusercontent.com/58855099/240291609-5b3711d6-7c1b-4656-882e-5de9b488d395.png)


### Install
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="S2JbPtGVRVMo" outputId="b674e44e-6fb9-4129-c90d-22c9903313c4"
!pip install 'finetuner[full]'
```

<!-- #region id="IRctQj4-zF9V" -->
## Prepare Synthesis Data
To perform synthesis, we need a query dataset and a corpus dataset, with the query dataset containing examples of user queries, and the corpus containing example search results.

We'll be generating training data based on the electronics section of the [Amazon cross-market dataset](https://xmrec.github.io/data/us/), a collection of products, ratings and reviews taken from Amazon. For our purposes, we will only be using the product names.  

We use the `xmarket_queries_da` and `xmarket_corpus_da` datasets, which we have already pre-processed and made available on the Jina AI Cloud. You can access them using `DocumentArray.pull`:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 203, "referenced_widgets": ["3cb81c3f5e0040aea02951eff9dff059", "40dd987eb8b14cff8dfc13bf93fa33c7", "75801a01473049c3ab4ae65c06b9a561", "a9c3e12ee1174a28b3931de1a82e2b4f", "9c7d39eb1cee431a841853aecf3f3591", "97d3f809f7c84ec3bb17a3066cbc2e73", "5953d68b80f74ecb99194d51f9e53a63", "15e315f55a1e472185b5704945661a01", "686e2843a27f4d68a93283d3ae936980", "9f8f7c6766d3453590cb6fc0733c55f2", "5c5904d26a0f4cb281f7105a23c5b294", "607aa3662ad4462f9ac343109c19ce3f", "86dbe530bc124cb199aacce49452ef44", "3a9d4001f2554c72b7d2daf12123ed9a", "4303b34cb255476f8b28ea5f10466308", "04a7972e4a794727899c46c6739f7d3b", "ee475535574e4479a88069861e009dae", "34698b70cdd04d5a912c0abccb65ecc0", "2ddcb79caeaa4c168017e6f16bcde4a7", "cc20600b050c4975b7e54a5efc7c7a07", "b16e4a2f26834841ade7013f850f4148", "03a2d6bcb651480f963e130023b2bdef", "87f9082328584446854bad3159a6e555", "ebc2f28f5ca44c34a06b452cb1f08e41", "7fede4f4efea4b3181dbf1b3c2612f82", "42773c4d708a44c597072a4f4a7fec99", "86441a78eb6c4e5682e5f866d8e08540", "22eaa7a2895d4f83b209f8351edc3c6b", "426d2897698b437d8e28040d8da96e3a", "fd88174b1c3549eebbfaff6d3c2c64f5", "4d1460d6bbb445ee8a793095389d2ca4"]} id="Srywu6C3YB0c" outputId="c5efed4e-d825-438b-f5b7-89547d01c84d"
import finetuner
from docarray import Document, DocumentArray

finetuner.login(force=True)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 418} id="hupAvfrwXJFk" outputId="96f66b12-8bec-4165-ed23-3166febac439"
query_data = DocumentArray.pull('finetuner/xmarket_queries_da')
corpus_data = DocumentArray.pull('finetuner/xmarket_corpus_da')

query_data.summary()
query_data[0].summary()
```

<!-- #region id="Xv1Qz1Q3mYu1" -->
The format of the data in these `DocumentArray`s is very simple, each `Document` wraps a single item, contained in its `text` field.
<!-- #endregion -->

<!-- #region id="pLoVzibX6BB8" -->
### Choosing models
Data synthesis jobs require two different models: a relation miner and a cross encoder.  

The relation miner is used to identify one similar and several dissimilar documents from the corpus data for each query in the query data.  

The cross encoder is then used to calculate a similarity between each query and its corresponding (dis)similar documents.  

Currently, we only support synthesis jobs for data in English, so when choosing a model you can just provide the `synthesis_model_en` object which contains the appropriate models for each of these tasks.
<!-- #endregion -->

<!-- #region id="KXtNctnH50AI" -->
## Start Synthesis Run
Now that we have the query and corpus datasets loaded as `DocumentArray`s, we can begin our synthesis run. We only need to provide the query and corpus data and the models that we are using.  

The `num_relations` parameter is set to 10. This parameter determines how many documents are retrieved for each query. There will always be one similar document and `(num_relations - 1)` dissimilar documents retrieved. These dissimilar documents are what make up the generated documents, so the size of the generated `DocumentArray` is always equal to `len(query_data) * (num_relations - 1)`. By default this parameter is set to 3, meaning that the size of the generated dataset would be twice as large as the size of the query dataset.
<!-- #endregion -->

```python id="7_EmudwyZlCO"
from finetuner.model import synthesis_model_en

synthesis_run = finetuner.synthesize(
    query_data='finetuner/xmarket_queries_da',
    corpus_data='finetuner/xmarket_corpus_da',
    models=synthesis_model_en,
    num_relations=10,
)

```

<!-- #region id="93yAUv4q-FQO" -->
### Monitoring

Now that we've created a run, we can check its status. You can monitor the run's progress with the function `synthesis_run.status()`, and the logs with `synthesis_run.logs()` or `synthesis_run.stream_logs()`. 

*Note: The job will take around 15 minutes to finish.*
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 1000, "referenced_widgets": ["fa63c575dc1f4118bc4b281cbf66ff78", "8725d03b98c34a7cad801951153b91ba"]} id="bZWaP1hbiA-g" outputId="0583be30-92ca-4b52-b756-2225c80525c1"
for entry in synthesis_run.stream_logs():
  print(entry)
```

<!-- #region id="wZL1O-YK-8kG" -->
Dependending on the size of the training data, some runs might take up to several hours. You can easily reconnect to your run later to monitor its status.

```python
import finetuner

finetuner.login()
synthesis_run = finetuner.get_run('my-synthesis-run')
print(f'Run status: {run.status()}')
```
<!-- #endregion -->

<!-- #region id="DoOuKaDU_F8U" -->
### Retrieving the data

Once the synthesis run has finished, the synthesised data will be pushed to the Jina AI Cloud under your account. The name of the pushed `DocumentArray` will be stored in `synthesis_run.train_data`.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 348} id="i6iiKEf7nyMM" outputId="dc39bb81-ffc5-4566-c351-494aa94998de"
train_data_name = synthesis_run.train_data
train_data = DocumentArray.pull(train_data_name)
train_data.summary()
```

<!-- #region id="cisFVD3o_bx3" -->
## Start Training with Synthesised Data

Using your synthesised data, you can now train a model using the `MarginMSELoss` function.  

 We have prepared the index and query datasets `xmarket-gpl-eval-queries` and `xmarket-gpl-eval-queries` so that we can evaluate the improvement provided by training on this data:
<!-- #endregion -->

```python id="ebfxt4NStvvg"
from finetuner.callback import EvaluationCallback

training_run = finetuner.fit(
    model='sbert-base-en',
    train_data=synthesis_run.train_data,
    loss='MarginMSELoss',
    optimizer='Adam',
    learning_rate=1e-5,
    epochs=3,
    callbacks=[
        EvaluationCallback(
            query_data='finetuner/xmarket-gpl-eval-queries',
            index_data='finetuner/xmarket-gpl-eval-index',
            batch_size=32,
        )
    ]
)
```

<!-- #region id="ubApI8OxARz3" -->
Just as before, you can monitor the progress of your run using `training_run.stream_logs()`:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} id="5tXpHElN4zzg" outputId="e2ed9587-c6e1-43da-da49-8db4d1d82c8d"
for entry in training_run.stream_logs():
  print(entry)
```

<!-- #region id="UcB3Fyk5Ao6T" -->
### Evaluating

Our `EvaluationCallback` during fine-tuning ensures that after each epoch, an evaluation of our model is run. We can access the evaluation results in the logs using `print(training_run.logs())`:

```bash
Training [3/3] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 470/470 0:00:00 0:02:34 • loss: 5.191
INFO     Done ✨                                                                              __main__.py:192
DEBUG    Finetuning took 0 days, 0 hours 11 minutes and 55 seconds                            __main__.py:194
INFO     Metric: 'sentence-transformers/msmarco-distilbert-base-v3_precision_at_k' before     __main__.py:207
fine-tuning:  0.16069 after fine-tuning: 0.19134
INFO     Metric: 'sentence-transformers/msmarco-distilbert-base-v3_recall_at_k' before        __main__.py:207
fine-tuning:  0.29887 after fine-tuning: 0.34635
INFO     Metric: 'sentence-transformers/msmarco-distilbert-base-v3_f1_score_at_k' before      __main__.py:207
fine-tuning:  0.13676 after fine-tuning: 0.16519
INFO     Metric: 'sentence-transformers/msmarco-distilbert-base-v3_hit_at_k' before           __main__.py:207
fine-tuning:  0.64277 after fine-tuning: 0.66069
INFO     Metric: 'sentence-transformers/msmarco-distilbert-base-v3_average_precision' before  __main__.py:207
fine-tuning:  0.34337 after fine-tuning: 0.39265
INFO     Metric: 'sentence-transformers/msmarco-distilbert-base-v3_reciprocal_rank' before    __main__.py:207
fine-tuning:  0.39998 after fine-tuning: 0.44711
INFO     Metric: 'sentence-transformers/msmarco-distilbert-base-v3_dcg_at_k' before           __main__.py:207
fine-tuning:  1.49618 after fine-tuning: 1.77899
INFO     Building the artifact ...                                                            __main__.py:231
INFO     Pushing artifact to Jina AI Cloud ...                                                __main__.py:260
```

The amount of improvement is highly dependent on the amount of data generated during synthesis, **as the amount of training data increases, so will the performance of the finetuned model**. To increase the number of documents generated, we can either increase the size of the query dataset provided to the `finetuner.synthesize` function, or increase value of the `num_relations` parameter, which will result in more documents being generated per query. Conversely, choosing a smaller value for `num_relations` would result in shorter generation and training times, but less improvement after training.  
To better understand the relationship between the amount of training data and the increase in performance, have a look at the [how much data?](https://finetuner.jina.ai/advanced-topics/budget/) section of our documentation.

<!-- #endregion -->
