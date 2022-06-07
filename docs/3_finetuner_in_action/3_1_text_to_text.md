# Text to text search using Bert

Searching large amounts of text documents with text queries is a very popular use-case, so of course Finetuner enables you to accomplish this easily.

This guide will lead you through an example use-case to show you how Finetuner can be used for text-to-text retrieval.

In Finetuner, two models are supported as backbones, namely `bert-base-cased` and `sentence-transformers/msmarco-distilbert-base-v3`, both of which are models hosted on Hugging Face.

In this example, we will fine-tune `sentence-transformers/msmarco-distilbert-base-v3` on the [Quora Question Pairs dataset](https://www.kaggle.com/competitions/quora-question-pairs), where the search task involves finding duplicate questions in the dataset. An example query for this search task might look as follows:

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

We will use Bert as an embedding model that embeds texts in a high dimensional space. We can fine-tune Bert so that questions that are duplicates of each other are represented in close proximity and questions that are not duplicates will have representations that are further apart in the embedding space. In this way, we can rank the embeddings in our search space by their proximity to the query question and return the highest ranking duplicates.


## Quora Dataset

We will use the Quora Question Pairs dataset to show-case Finetuner for text-to-text search. We have already pre-processed these and made them available for you to pull from hubble. Do this as follows:

```python
from docarray import DocumentArray

train_data = DocumentArray.pull('quora_train.da')
query_data = DocumentArray.pull('quora_query_dev.da')
index_data = DocumentArray.pull('quora_index_dev.da')

train_data.summary()
```

Your `train_data` `DocumentArray` summary should look like this:

```python
╭──────────────── Documents Summary ────────────────╮
│                                                   │
│   Length                 104598                   │
│   Homogenous Documents   True                     │
│   Common Attributes      ('id', 'text', 'tags')   │
│   Multimodal dataclass   False                    │
│                                                   │
╰───────────────────────────────────────────────────╯
╭───────────────────── Attributes Summary ─────────────────────╮
│                                                              │
│   Attribute   Data type   #Unique values   Has empty value   │
│  ──────────────────────────────────────────────────────────  │
│   id          ('str',)    104598           False             │
│   tags        ('dict',)   104598           False             │
│   text        ('str',)    104559           False             │
│                                                              │
╰──────────────────────────────────────────────────────────────╯
```

So we have 104598 training `Documents`. Each `Document` consists of a text field that contains the question, as well as a `finetuner_label` which indicates the label to which the question belongs. If multiple questions have the same label, they are duplicates of one another. If they have different `finetuner_label`s, they are not duplicates of each other.

As for the evaluation dataset, we load `query_data` and `index_data` seperately. The `query_data` has the same structure as the `train_data`, consisting of labelled documents. The `index_data` is the data against which the queries will be matched, and contains many documents, some of which may be irrelevant to the queries (ie. they have no duplicated in the `query_data`).
If you look at the summaries for the `query_data` and `index_data`, you will find that they have the following number of instances:

```
Length of queries DocumentArray: 5000
Length of index DocumentArray: 15746
```


## Finetune the Bert model

Now that we have the training and evaluation datasets loaded as `DocumentArray`s, we can start our fine-tuning experiment with this dataset using Bert.

```python
import finetuner
from finetuner.client.callbacks import EvaluationCallback

# Start fine-tuning as a run within an experiment
finetuner.fit(
    model='sentence-transformers/msmarco-distilbert-base-v3',
    train_data=train_data,
    experiment_name='finetune-quora-dataset',
    run_name='finetune-quora-dataset-distilbert-1',
    description='this is a trial run on quora dataset with msmarco-distilbert-base-v3.',
    loss='TripletMarginLoss',
    miner='TripletMarginMiner',
    optimizer='Adam',
    learning_rate = 1e-4,
    epochs=3,
    batch_size=128,
    scheduler_step='batch',
    freeze=False, # We are training the whole bert model, not an additional MLP.
    output_dim=None,
    multi_modal=False, # we only have textual data
    image_modality=None,
    text_modality=None,
    cpu=False,
    num_workers=4,
    callbacks=[EvaluationCallback(query_data=query_data, index_data=index_data, batch_size=128)]
)
```

You can check the status of your experiment and save the model when the experiment has completed.

```python
experiment = finetuner.get_experiment('finetune-quora-dataset')
run = experiment.get_run('finetune-quora-dataset-distilbert-1')
print(f'Run status: {run.status()}')
print(f'Run logs: {run.logs()}')

run.save_model('.')
```