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

<!-- #region id="Huf1E2zq7JWb" -->
# Text-to-Text Search via BERT

<a href="https://colab.research.google.com/drive/1Ui3Gw3ZL785I7AuzlHv3I0-jTvFFxJ4_?usp=sharing"><img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg"></a>

Searching large amounts of text documents with text queries is a very popular use case and Finetuner enables you to accomplish this easily.

This guide will lead you through an example use case to show you how Finetuner can be used for text-to-text retrieval (Dense Retrieval).

*Note, please consider switching to GPU/TPU Runtime for faster inference.*

## Install
<!-- #endregion -->

```python id="CSuWo72R7Sno"
!pip install 'finetuner[full]'
```

<!-- #region id="FPDhvWkw7kas" -->
## Task

In Finetuner, two BERT models are supported as backbones, namely `bert-base-en` and `sbert-base-en`, both of which are models hosted on Hugging Face.

In this example, we will fine-tune `bert-base-en` on the [Quora Question Pairs](https://www.sbert.net/examples/training/quora_duplicate_questions/README.html?highlight=quora#dataset) dataset, where the search task involves finding duplicate questions in the dataset. An example query for this search task might look as follows:

```
How can I be a good geologist?

```

Retrieved documents that could be duplicates for this question should then be ranked in the following order:

```
What should I do to be a great geologist?
How do I become a geologist?
What do geologists do?
...

```

We can fine-tune BERT so that questions that are duplicates of each other are represented in close proximity and questions that are not duplicates will have representations that are further apart in the embedding space. In this way, we can rank the embeddings in our search space by their proximity to the query question and return the highest-ranking duplicates.
<!-- #endregion -->

<!-- #region id="SfR6g0E_8fOz" -->
## Data

We will use the [Quora Question Pairs](https://www.sbert.net/examples/training/quora_duplicate_questions/README.html?highlight=quora#dataset) dataset to showcase Finetuner for text-to-text search. We have already pre-processed this dataset and made it available for you to pull from Jina AI Cloud. Do this as follows:
<!-- #endregion -->

```python id="pwS11Nsg7jPM"
import finetuner
from finetuner import DocumentArray, Document

finetuner.login(force=True)

```

```python id="8PIO5T--p4tR"
train_data = DocumentArray.pull('finetuner/quora-train-da', show_progress=True)
query_data = DocumentArray.pull('finetuner/quora-test-query-da', show_progress=True)
index_data = DocumentArray.pull('finetuner/quora-test-index-da', show_progress=True)

train_data.summary()
```

<!-- #region id="r_IlEIp59g9v" -->
So we have 104598 training `Document`s. Each `Document` consists of a text field that contains the question, as well as a `finetuner_label` which indicates the label to which the question belongs. If multiple questions have the same label, they are duplicates of one another. If they have different `finetuner_label`s, they have no duplicates of each other.

As for the evaluation dataset, we load `query_data` and `index_data` separately. The `query_data` have the same structure as the `train_data`, consisting of labeled documents. The `index_data` are the data against which the queries will be matched, and contain many documents, some of which may be irrelevant to the queries (i.e. they have no duplicated in the `query_data`).
If you look at the summaries for the `query_data` and `index_data`, you will find that they have the following number of samples:

```
Length of queries DocumentArray: 5000
Length of index DocumentArray: 15746
```
<!-- #endregion -->

<!-- #region id="aXYrABkN9vYO" -->
## Backbone model
To keep things simple, we have decided to fine-tune the BERT model `bert-base-en`. We could also have chosen `sbert-base-en` as our base model, which has already been fine-tuned on the MSMarco dataset. 
However, for the purpose of this experiment, we want to explore how much improvement in performance we can gain from fine-tuning `bert-base-en` on the Quora Question Pairs dataset using Finetuner. 
Perhaps in the future, we might want to create another run where we experiment with fine-tuning other BERT models.


<!-- #endregion -->

<!-- #region id="IAlQArUB99oG" -->
## Fine-tuning

Now that we have the training and evaluation datasets loaded as `DocumentArray`s and selected our model, we can start our fine-tuning run.
<!-- #endregion -->

```python id="hsRfjf1Z8ymZ"
from finetuner.callback import EvaluationCallback

run = finetuner.fit(
    model='bert-base-en',
    train_data='finetuner/quora-train-da',
    loss='TripletMarginLoss',
    optimizer='Adam',
    learning_rate = 1e-5,
    epochs=3,
    batch_size=128,
    device='cuda',
    callbacks=[
        EvaluationCallback(
            query_data='finetuner/quora-test-query-da',
            index_data='finetuner/quora-test-index-da',
            batch_size=32
        )
    ]
)
```

<!-- #region id="j_MxAW9E-ddZ" -->
Our fine-tuning call has a lot of arguments. Let's discuss what the most important ones are responsible for. 

Most importantly, we select our model with `model='bert-base-en'` and pass our training data with `train_data=train_data`. These two arguments are required. 
We set our `experiment_name` to `'finetune-quora-dataset'` and our `run_name` to `'finetune-quora-dataset-bert-base-en'`. 
This will make it easy for us to retrieve the experiment and run in the future. We also provide a short description of our run, just for some extra context. 

For this run, we select Finetuner's `TripletMarginLoss` and `TripletMarginMiner`, as they are most relevant for our use-case. The `TripletMarginLoss` measures the similarity between three tensors, namely the anchor, a positive sample and a negative sample. This makes sense for our task, since we want duplicate questions to have representations closer together, while non-duplicates should have more dissimilar representations. Likewise, the `TripletMarginMiner` outputs a tuple of size 3, with an anchor, a positive sample and a negative sample.

Lastly, we provide an `EvaluationCallback` with our `query_data` and `index_data`. This evaluation is done at the end of each epoch and its results will be visible to us in the logs, which we will monitor in the next section. Since we have not specified which metrics should be applied, default metrics will be computed. The `Evaluation` section of this guide will show you the default metrics.
<!-- #endregion -->

<!-- #region id="h0DGNRo8-lZD" -->
## Monitoring

Now that we've created a run, let's see its status. You can monitor the run by checking the status - `run.status()`, -and the logs `run.logs()` or `run.stream_logs()`. 

*note, the job will take around 15 minutes to finish.*
<!-- #endregion -->

```python id="gajka0TG-S6u"
for entry in run.stream_logs():
    print(entry)
```

<!-- #region id="7AuB0IWC_CSt" -->
Dependending on the size of the training data, some runs might take up to several hours. You can later reconnect to your run very easily to monitor its status.

```python
import finetuner

finetuner.login()
run = finetuner.get_run('finetune-quora-dataset-bert-base-en')
print(f'Run status: {run.status()}')
```
<!-- #endregion -->

<!-- #region id="agqrb0TX_Y4b" -->
## Evaluating

Our `EvaluationCallback` during fine-tuning ensures that after each epoch, an evaluation of our model is run. We can access the evaluation results in the logs as follows `print(run.logs())`:

```bash
  Training [3/3] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 818/818 0:00:00 0:03:05 • loss: 0.000
[15:36:40] DEBUG    Metric: 'model_average_precision' Value: 0.95728                                     __main__.py:202
           DEBUG    Metric: 'model_dcg_at_k' Value: 1.33912                                              __main__.py:202
           DEBUG    Metric: 'model_f1_score_at_k' Value: 0.13469                                         __main__.py:202
           DEBUG    Metric: 'model_hit_at_k' Value: 0.99720                                              __main__.py:202
           DEBUG    Metric: 'model_ndcg_at_k' Value: 0.97529                                             __main__.py:202
           DEBUG    Metric: 'model_precision_at_k' Value: 0.07653                                        __main__.py:202
           DEBUG    Metric: 'model_r_precision' Value: 0.94393                                           __main__.py:202
           DEBUG    Metric: 'model_recall_at_k' Value: 0.99301                                           __main__.py:202
           DEBUG    Metric: 'model_reciprocal_rank' Value: 0.96686                                       __main__.py:202
           INFO     Done ✨                                                                              __main__.py:204
           INFO     Saving fine-tuned models ...                                                         __main__.py:207
           INFO     Saving model 'model' in /usr/src/app/tuned-models/model ...                          __main__.py:218
[15:36:41] INFO     Pushing saved model to Jina AI Cloud ...                                                    __main__.py:225
[15:37:32] INFO     Pushed model artifact ID: '62b9cb73a411d7e08d18bd16'                                 __main__.py:231
           INFO     Finished 🚀                                                                          __main__.py:233                                                  __main__.py:225
```
<!-- #endregion -->

<!-- #region id="KTfBfB8A_1fO" -->
## Saving
Once your run has successfully completed, you can save your fine-tuned model in the following way:
<!-- #endregion -->

```python id="z7AJw3X9-7C-"
artifact = run.save_artifact('bert-model')
```

<!-- #region id="YYgPIR_kAI6z" -->
## Inference

Now you saved the `artifact` into your host machine,
let's use the fine-tuned model to encode a new `Document`:

```{admonition} Inference with ONNX
In case you set `to_onnx=True` when calling `finetuner.fit` function,
please use `model = finetuner.get_model(artifact, is_onnx=True)`
```
<!-- #endregion -->

```python id="Qs2G-rNFAJ4I"
model = finetuner.get_model(artifact=artifact, device='cuda')

query = DocumentArray([Document(text='How can I be an engineer?')])

finetuner.encode(model=model, data=query)
finetuner.encode(model=model, data=index_data)
assert query.embeddings.shape == (1, 768)
```

<!-- #region id="a_vUDidVIkh7" -->
And finally, you can use the embedded `query` to find top-k semantically related text within `index_data` as follows:
<!-- #endregion -->

```python id="-_bM-TXRE2h7"
query.match(index_data, limit=10, metric='cosine')
```

<!-- #region id="53Xtm0hidrjs" -->
## Before and after
We can directly compare the results of our fine-tuned model with its zero-shot counterpart to get a better idea of how finetuning affects the results of a search. While the zero-shot model is able to produce results that are very similar to the initial query, it is common for the topic of the question to change, with the structure staying the same. After fine-tuning, the returned questions are consistently relevant to the initial query, even in cases where the structure of the sentence is different.

```plaintext
Query: What's the best way to start learning robotics?
 matches pretrained:
 - What is the best way to start with robotics?
 - What is the best way to learn web programming?
 - What is the best way to start learning Japanese from scratch?
 - What is the best way to start learning a language?
 - What is the best place to learn data science?
 matches finetuned
 - What is good way to learn robotics?
 - What is the best way to start with robotics?
 - How can I get started learning about robotics?
 - How can I start to learn robotics from zero?
 - From where should a complete beginner (0 knowledge) start in learning robotics?

Query: What online platforms can I post ads for beer money opportunity?
 matches pretrained:
 - On what online platforms can I post ads for beer money opportunity?
 - How can I restore a mobile-number only Facebook messenger account?
 - How do I earn money online without any blog or website?
 - Do I need to register my self to selling products on online platforms in India?
 - Which is the best website where we can buy instagram followers and likes?
 matches finetuned
 - On what online platforms can I post ads for beer money opportunity?
 - What are some legit ways to earn money online?
 - What are some genuine ways to earn money online?
 - What are the best legitimate methods to making money online?
 - What are the legitimate ways to earn money online?

```

<!-- #endregion -->

<!-- #region id="czK5pSUEAcdS" -->
That's it!
<!-- #endregion -->
