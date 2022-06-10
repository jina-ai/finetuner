(text-to-text)=
# Text to text search using Bert

Searching large amounts of text documents with text queries is a very popular use-case, so of course Finetuner enables you to accomplish this easily.

This guide will lead you through an example use-case to show you how Finetuner can be used for text to text retrieval.


## Task Overview

In Finetuner, two Bert models are supported as backbones, namely `bert-base-cased` and `sentence-transformers/msmarco-distilbert-base-v3`, both of which are models hosted on Hugging Face.

In this example, we will fine-tune `bert-base-cased` on the [Quora Question Pairs](https://www.sbert.net/examples/training/quora_duplicate_questions/README.html?highlight=quora#dataset) dataset, where the search task involves finding duplicate questions in the dataset. 
An example query for this search task might look as follows:

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

```{admonition} See Also: Model and dataset info
:class: seealso
If you'd like more information about the Bert model we are fine-tuning please visit [Hugging Face](https://huggingface.co/bert-base-cased). 
Supplementary information about the Quora Question Pairs dataset can be found on the [Sentence-Transformers](https://www.sbert.net/examples/training/quora_duplicate_questions/README.html?highlight=quora#dataset) website.
```

## Prepraring data

We will use the [Quora Question Pairs](https://www.sbert.net/examples/training/quora_duplicate_questions/README.html?highlight=quora#dataset) dataset to show-case Finetuner for text to text search. We have already pre-processed this dataset and made it available for you to pull from hubble. Do this as follows:

```python
from docarray import DocumentArray

train_data = DocumentArray.pull('quora_train.da')
query_data = DocumentArray.pull('quora_query_dev.da')
index_data = DocumentArray.pull('quora_index_dev.da')

train_data.summary()
```

Your `train_data` `DocumentArray` summary should look like this:

```bash
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

So we have 104598 training `Document`s. Each `Document` consists of a text field that contains the question, as well as a `finetuner_label` which indicates the label to which the question belongs. If multiple questions have the same label, they are duplicates of one another. If they have different `finetuner_label`s, they are not duplicates of each other.

As for the evaluation dataset, we load `query_data` and `index_data` seperately. The `query_data` has the same structure as the `train_data`, consisting of labelled documents. The `index_data` is the data against which the queries will be matched, and contains many documents, some of which may be irrelevant to the queries (ie. they have no duplicated in the `query_data`).
If you look at the summaries for the `query_data` and `index_data`, you will find that they have the following number of instances:

```
Length of queries DocumentArray: 5000
Length of index DocumentArray: 15746
```

## Choosing the model
To keep things simple, we have decided to fine-tune the Bert model `bert-base-cased`. We could also have chosen `sentence-transformers/msmarco-distilbert-base-v3` as our base model, which has already been fine-tuned on the MSMarco dataset. 
However, for the purpose of this experiment, we want to explore how much improvement in performance we can gain from fine-tuning `bert-base-cased` on the Quora Question Pairs dataset using Finetuner. 
Perhaps in the future, we might want to create another run where we experiment with fine-tuning other Bert models.

```{admonition} Backbones
:class: tip
 Finetuner also supports fine-tuning other models, see the {ref}`choose backbone <choose-backbone>` section.
 ```


## Creating a fine-tuning run

Now that we have the training and evaluation datasets loaded as `DocumentArray`s and selected our model, we can start our fine-tuning run.

```python
import finetuner
from finetuner.callback import EvaluationCallback

# Make sure to login to Jina Cloud
finetuner.login()

# Start fine-tuning as a run within an experiment
finetuner.fit(
    model='bert-base-cased',
    train_data=train_data,
    experiment_name='finetune-quora-dataset',
    run_name='finetune-quora-dataset-bert-base-cased',
    description='this is a trial run on quora dataset with bert-base-cased.',
    loss='TripletMarginLoss',
    miner='TripletMarginMiner',
    optimizer='Adam',
    learning_rate = 1e-5,
    epochs=3,
    batch_size=128,
    scheduler_step='batch',
    freeze=False, # We are training the whole bert model, not an additional MLP.
    output_dim=None,
    cpu=False,
    num_workers=4,
    callbacks=[EvaluationCallback(query_data=query_data, index_data=index_data, batch_size=256)]
)
```

Our fine-tuning call has a lot of arguments. Let's discuss what the most important ones are responsible for. 

Most importantly, we select our model with `model='bert-base-cased'` and pass our training data with `train_data=train_data`. These two arguments are required. 
We set our `experiment_name` to `'finetune-quora-dataset'` and our `run_name` to `'finetune-quora-dataset-bert-base-cased'`. 
This will make it easy for us to retrieve the experiment and run in the future. We also provide a short description of our run, just for some extra context. 

For this run, we select Finetuner's `'TripletMarginLoss'` and `'TripletMarginMiner'`, as they are most relevant for our use-case. The `'TripletMarginLoss'` measures the similarity between three tensors, namely the anchor, a positive sample and a negative sample. This makes sense for our task, since we want duplicate questions to have representations closer together, while non-duplicates should have more dissimilar representations. Likewise, the `'TripletMarginMiner'` outputs a tuple of size 3, with an anchor, a positive sample and a negative sample.

```{admonition} See Also: TripletMarginLoss and TripletMarginMiner
:class: seealso
More information about `TripletMarginLoss` and `TripletMarginMiner` can be found in the [PyTorch](https://pytorch.org/docs/stable/generated/torch.nn.TripletMarginLoss.html) and [metric learning](https://kevinmusgrave.github.io/pytorch-metric-learning/miners/#tripletmarginminer) documentation respectively.
```

Lastly, we provide an `EvaluationCallback` with our `query_data` and `index_data`. This evaluation is done at the end of each epoch and its results will be visible to us in the logs, which we will monitor in the next section. Since we have not specified which metrics should be applied, default metrics will be computed. The `Evaluation` section of this guide will show you the default metrics.


## Monitoring your runs
Now that we've created a run, let's see its status. You can monitor the run by checking the status - `run.status()` or the logs - `run.logs()`. 
```python
print(run.status())
```

```bash
{'status': 'CREATED', 'details': 'Run submitted and awaits execution'}
```

Since some runs might take up to several hours/days, you can reconnect to your run very easily to monitor its status.
```python
import finetuner

finetuner.login()
experiment = finetuner.get_experiment('finetune-quora-dataset')
run = experiment.get_run('finetune-quora-dataset-bert-base-cased')
print(f'Run status: {run.status()}')
```


## Save your model
Once your run has successfully completed, you can save your fine-tuned model in the following way:
```python
run.save_model('finetune-quora-dataset-bert-base-cased')
```

## Evaluation

Our `EvaluationCallback` during fine-tuning ensures that after each epoch, an evaluation of our model is run. We can access the evaluation results in the logs as follows:

```python
import finetuner

finetuner.login()
experiment = finetuner.get_experiment('finetune-quora-dataset')
run = experiment.get_run('finetune-quora-dataset-bert-base-cased')
print(f'Run logs: {run.logs()}')
```

```bash
(log output)
```