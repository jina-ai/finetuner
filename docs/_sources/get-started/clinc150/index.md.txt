# Finetuning a Transformer for Intent Classification

````{info}
This example builds a transformer model using `pytorch` and the `transformers` package. You can install both packages using:

```
pip install torch transformers
```
````

This example demonstrates how to finetune a model on textual data using `finetuner`.
Specifically we will tune a transformer model on an Intent Classification task. Intent
classification is the problem where we try to predict the user intent from a user utterance.
It is a common step in chatbots and Conversational AI, where after the user speech has been
decoded to text, we try to represent the meaning of the text symbolically by predicting
intents and semantic entities. For example:

* I want to book a flight - intent: `book-flight`
* What is the weather forecast for tomorrow? - intent: `get-weather`

The intent classification task is usually formulated as text classification i.e. we build
a classifier to predict intents on input text. In this example, we will formulate
the problem as a search task and use `finetuner` to tune text representations.

We will build an embedding model that embeds text to a high dimensional space and then
we will tune the model so that texts that belong to the same class (intent) are
represented in proximity and texts that belong to separate classes are pulled apart in
our embedding space. To convert our embedding model back to a useful intent prediction
model, we will implement a simple nearest neighbor rule.


## CLINC150

We will use the CLINC150 dataset as the base of our experiment. It is a dataset of
utterance-intent pairs and is commonly used for evaluating intent models. It comes
in train, val and test splits and contains 150 intents from various chatbot domains.

CLINC150 comes in different sizes with regards to the number of utterances per intent,
to facilitate experimentation on few-shot learning methods. We will use the full version
which includes 100, 20 and 30 utterances per intent in the train, val and test
splits respectively.

For more info on the CLINC150 dataset, check out the
[dataset repo](https://github.com/clinc/oos-eval).

Firstly, let's dowload the dataset:
```bash
curl -o data_full.json https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json
```

Let's look at some examples:

```json
{
  "examples": [
    [
      "how much is $1 usd in euros", 
      "exchange_rate"
    ],
    [
      "what am i listening to right now", 
      "what_song"
    ],
    [
      "can you check if meeting rooms are available between 4 and 5", 
      "schedule_meeting"
    ], 
    [
      "you know procedure to cook apple pie", 
      "recipe"
    ]
  ]
}
```

The dataset is a JSON file with utterance intent pairs. We convert the train, val and
test splits to `DocumentArray`s and attach the intent label for each doc in 
`doc.tags['finetuner_label']`.

```python
import json

from docarray import Document, DocumentArray

DATASET_PATH = 'data_full.json'

with open(DATASET_PATH, 'r') as f:
    data = json.load(f)

train_data = DocumentArray(
    [
        Document(text=utterance, tags={'finetuner_label': intent})
        for utterance, intent in data['train']
    ]
)
val_data = DocumentArray(
    [
        Document(text=utterance, tags={'finetuner_label': intent})
        for utterance, intent in data['val']
    ]
)
test_data = DocumentArray(
    [
        Document(text=utterance, tags={'finetuner_label': intent})
        for utterance, intent in data['test']
    ]
)
```
```
Num train samples: 15000
Num val samples: 3000
Num test samples: 4500
```


## Embedding model

As described above, we will use `finetuner` to finetune an embedding model in order
to bring representations of the same intent, closer in the embedding space. For that
we will use the `transformers` library to define a transformer-based embedding model.
We will load a pre-trained transformer as our starting point,
the [paraphrase-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L6-v2)
model from [sentence-transformers](https://www.sbert.net/index.html).

```python
import torch
from transformers import AutoModel

TRANSFORMER_MODEL = 'sentence-transformers/paraphrase-MiniLM-L6-v2'

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )

class TransformerEmbedder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained(TRANSFORMER_MODEL)

    def forward(self, inputs):
        out = self.model(**inputs)
        return mean_pooling(out, inputs['attention_mask'])
```

Our model is a pre-trained embedding model, i.e. it outputs a high-dimensional
representation given some input text and it has been pre-trained on large text corpora.

To use the model defined above, we need to be able to convert raw text to the tensor
format that our model accepts as input. To do that, we need to use the BPE tokenizer,
provided by the `transformers` package, that converts texts to BPE encoded arrays that
our model accepts as input.

We make use of the collate function that `finetuner` supports. The collate function
offers a way to specify the conversion of batch elements to model input tensors.

```python
from typing import List

from transformers import AutoTokenizer

MAX_SEQ_LEN = 50
tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL)

def collate_fn(inputs: List[str]):
    return tokenizer(
        inputs,
        truncation=True,
        max_length=MAX_SEQ_LEN,
        padding=True,
        return_tensors='pt',
    )
```


## Fine-tuning

We will finetune the model for 6 epochs using a batch size of 256. We are using a learning
rate of 1e-4, with the AdamW optimizer and a linear learning rate scheduler with warmup.
This weight update strategy is often recommended for finetuning transformer models.
Finetuner allows us to configure the optimizer and the scheduler, via the
`configure_optimizer` argument which should be a function that accepts the model as input
and returns the optimizer and the scheduler as a tuple.

For the training objective, we are using the `TripletLoss` in conjunction with the
`TripletEasyHardMiner` with easy positive and hard negative strategies and a margin of `0.4`.

Finally we make use of the various callbacks provided by `finetuner`, to inject
functionalities in the training loop. Specifically, we use the `EvaluationCallback`
so that metrics are computed on the val set after each epoch, the `EarlyStopping`
callback which monitors the average precision to trigger early stopping if the metric stops
increasing, the `BestModelCheckpoint` that saves the best performing model (in terms of
average precision) every epoch and finally the `WandBLogger` callback that logs our
training information using [Weights and Biases](https://wandb.ai/site).

To use the weights and biases logger, you should install the `wandb` client and login,
provided you have an active account:

```bash
pip install wandb
wandb login
```

Let's start fine-tuning!

```python
import math

import finetuner
from finetuner.tuner.callback import (
    BestModelCheckpoint,
    EarlyStopping,
    EvaluationCallback,
    WandBLogger,
)
from finetuner.tuner.pytorch.losses import TripletLoss
from finetuner.tuner.pytorch.miner import TripletEasyHardMiner
from transformers.optimization import get_linear_schedule_with_warmup

EPOCHS = 6
BATCH_SIZE = 256
LEARNING_RATE = 1e-4
NUM_WORKERS = 8
NUM_ITEMS_PER_CLASS = 4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def configure_optimizer(model: torch.nn.Module):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=50,
        num_training_steps=EPOCHS * math.ceil(len(train_data) / BATCH_SIZE),
    )
    return optimizer, scheduler

evaluation_callback = EvaluationCallback(val_data, limit=20, num_workers=NUM_WORKERS)
wandb_logger = WandBLogger()
early_stopping = EarlyStopping(patience=1, monitor='average_precision')
best_model_ckpt = BestModelCheckpoint(
    save_dir='checkpoints', monitor='average_precision'
)

finetuned_model = finetuner.fit(
    TransformerEmbedder(),
    train_data=train_data,
    eval_data=val_data,
    loss=TripletLoss(
        distance='cosine',
        margin=0.4,
        miner=TripletEasyHardMiner(pos_strategy='easy', neg_strategy='hard'),
    ),
    configure_optimizer=configure_optimizer,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    num_items_per_class=NUM_ITEMS_PER_CLASS,
    device=DEVICE,
    collate_fn=collate_fn,
    callbacks=[evaluation_callback, wandb_logger, early_stopping, best_model_ckpt],
)
```

Let's go through the weights and biases run for various training stats. Below is our
learning rate schedule, our training and validation loss and some evaluation metrics
calculcated in our val split:

```{figure} wandb01.png
```
```{figure} wandb02.png
```

Now it's time to see how much we improved. To evaluate the model, we can use the
built-in `Evaluator` component of `finetuner` that allows us to compute information
retrieval metrics. We will evaluate both the pre-trained and the fine-tuned model
on the test split of our dataset:

```python
from finetuner.tuner.evaluation import Evaluator

pretrained_model = TransformerEmbedder()
evaluator = Evaluator(test_data, embed_model=pretrained_model)
pretrained_metrics = evaluator.evaluate(
    limit=30,
    num_workers=NUM_WORKERS,
    batch_size=BATCH_SIZE,
    device=DEVICE,
    collate_fn=collate_fn,
)

evaluator = Evaluator(test_data, embed_model=finetuned_model)
finetuned_metrics = evaluator.evaluate(
    limit=30,
    num_workers=NUM_WORKERS,
    batch_size=BATCH_SIZE,
    device=DEVICE,
    collate_fn=collate_fn,
)
```
The evaluation metrics are presented in the table below:

| metrics           | pre-trained | fine-tuned |
|:-----------------:|:-----------:|:----------:|
| r_precision       | 0.660       | **0.915**  |
| precision_at_k    | 0.592       | **0.882**  |
| recall_at_k       | 0.592       | **0.882**  |
| f1_score_at_k     | 0.592       | **0.882**  |
| average_precision | 0.818       | **0.950**  |
| hit_at_k          | **0.996**   | 0.992      |
| reciprocal_rank   | 0.934       | **0.971**  |
| dcg_at_k          | 6.552       | **8.841**  |
| ndcg_at_k         | 0.909       | **0.968**  |

The pre-trained model has a solid performance in our dataset. Using `finetuner` though,
we managed to gain a significant improvement in precision, recall and F1 score as
well as DCG and NDCG!


## Back to the classification task

We fine-tuned our embedding model and improved significantly in terms of IR metrics.
But how about intent accuracy? How many times do we predict the correct intent? What about
predicting intents in the first place?

In an intent classification task, we want to classify utterances to intents.
So far we formulated the task as a search problem, but to actually use the model
we need to revert back to the classification task. The missing part is a
decision function that can produce intent classes on a test utterance, by
utilising the finetuned embedding model.

A straight-forward way to do that, is to embed the test utterance using our
fine-tuned model and search for the nearest neighbor from a set of utterances
with pre-computed embeddings. This set of utterances with pre-computed
embeddings is usually refered to as the index, and we can use our training
data for it. The class that the nearest neighbor belongs to, will be the class
that we assign to the test utterance.

To go one step further, we can also choose to fetch multiple neighbors
to the test utterance and decide on the intents to return, using a simple rule
that takes into account both the class and the distance of each neighbor.

We implement this function, using `docarray`s `match` and `finetuner`s `embed`
methods.

```python
from collections import defaultdict
from typing import Tuple

from finetuner import embed


def predict_intents(
    utterance: str,
    model: torch.nn.Module,
    index: DocumentArray,
    k: int = 20,
) -> List[Tuple[str, float]]:
    """
    Find top k nearest neighbors in a search query
    and compute intents by aggregating distances
    """
    doc = Document(text=utterance)
    embed(
        DocumentArray(doc),
        embed_model=model,
        device=DEVICE,
        batch_size=1,
        collate_fn=collate_fn,
    )
    doc.match(index, limit=k)

    intents = [m.tags['finetuner_label'] for m in doc.matches]
    distances = [m.scores['cosine'].value for m in doc.matches]
    sum_distances = sum(distances)
    scores = [dist / sum_distances for dist in distances]

    output = defaultdict(float)
    for intent, score in zip(intents, scores):
        output[intent] += score

    return sorted(list(output.items()), key=lambda x: x[1], reverse=True)
```

Let's index our training data:

```python
from copy import deepcopy

pretrained_model = TransformerEmbedder()

pretrained_index = deepcopy(train_data)
finetuned_index = deepcopy(train_data)

embed(
    pretrained_index,
    embed_model=pretrained_model,
    device=DEVICE,
    batch_size=BATCH_SIZE,
    collate_fn=collate_fn,
)
embed(
    finetuned_index,
    embed_model=finetuned_model,
    device=DEVICE,
    batch_size=BATCH_SIZE,
    collate_fn=collate_fn,
)
```

Let's now use this method to run some test cases.

```python
utterance = 'Where do you think I should travel to this Christmas?'
intents_pretrained = predict_intents(utterance, pretrained_model, pretrained_index, k=1)
intents_finetuned = predict_intents(utterance, finetuned_model, finetuned_index, k=1)
```
| model `k=1` | Where do you think I should travel to this Christmas? |
|:-----------:|:-----------------------------------------------------:|
| pre-trained | ('next_holiday', 1.0)                                 |
| fine-tuned  | ('travel_suggestion', 1.0)                            |

Trying with `k=20`.
```python
utterance = 'Where do you think I should travel to this Christmas?'
intents_pretrained = predict_intents(utterance, pretrained_model, pretrained_index, k=20)
intents_finetuned = predict_intents(utterance, finetuned_model, finetuned_index, k=20)
```

| model `k=20`| Where do you think I should travel to this Christmas?                              |
|:-----------:|:----------------------------------------------------------------------------------:|
| pre-trained | ('next_holiday', 0.851), ('travel_suggestion', 0.103), ('spending_history', 0.046) |
| fine-tuned  | ('travel_suggestion', 1.0)                                                         |


What about utterances with 2 intents?
```python
utterance = 'What is my location right now? Can you share it with Dave?'
intents_pretrained = predict_intents(utterance, pretrained_model, pretrained_index, k=20)
intents_finetuned = predict_intents(utterance, finetuned_model, finetuned_index, k=20)
```

| model `k=20`| What is my location right now? Can you share it with Dave?  |
|:-----------:|:-----------------------------------------------------------:|
| pre-trained | ('current_location', 0.628), ('share_location', 0.372)      |
| fine-tuned  | ('share_location', 0.745), ('current_location', 0.255)      |

Since we have a way to predict classes in text data, we can evaluate the model
using classification metrics. Let's try to compare pre-trained and fine-tuned
models in terms of accuracy.

```python
true_intents = [doc.tags['finetuner_label'] for doc in test_data]
pretrained_predicted_intents = [
    predict_intents(doc.text, pretrained_model, pretrained_index, k=20)[0][0]
    for doc in test_data
]
finetuned_predicted_intents = [
    predict_intents(doc.text, finetuned_model, finetuned_index, k=20)[0][0]
    for doc in test_data
]

pretrained_acc = sum(
    [int(t == p) for t, p in zip(true_intents, pretrained_predicted_intents)]
) / len(test_data)

finetuned_acc = sum(
    [int(t == p) for t, p in zip(true_intents, finetuned_predicted_intents)]
) / len(test_data)
```

| model `k=20`| accuracy  |
|:-----------:|:---------:|
| pre-trained | 0.874     |
| fine-tuned  | **0.946** |


## Full tutorial

For reference, the full tutorial code is given in the snippet below.

````{dropdown} Complete source code
```python
import json
import math
from collections import defaultdict
from copy import deepcopy
from typing import List, Tuple

import torch
from docarray import Document, DocumentArray
from transformers import AutoModel, AutoTokenizer
from transformers.optimization import get_linear_schedule_with_warmup

import finetuner
from finetuner import embed
from finetuner.tuner.callback import (
    BestModelCheckpoint,
    EarlyStopping,
    EvaluationCallback,
    WandBLogger,
)
from finetuner.tuner.evaluation import Evaluator
from finetuner.tuner.pytorch.losses import TripletLoss
from finetuner.tuner.pytorch.miner import TripletEasyHardMiner


# ---- DATA ----------------------------------------------------------------------------

DATASET_PATH = 'data_full.json'

# Load the CLINC150 dataset
with open(DATASET_PATH, 'r') as f:
    data = json.load(f)

# Load train, val and test data in DocumentArray format
train_data = DocumentArray(
    [
        Document(text=utterance, tags={'finetuner_label': intent})
        for utterance, intent in data['train']
    ]
)
val_data = DocumentArray(
    [
        Document(text=utterance, tags={'finetuner_label': intent})
        for utterance, intent in data['val']
    ]
)
test_data = DocumentArray(
    [
        Document(text=utterance, tags={'finetuner_label': intent})
        for utterance, intent in data['test']
    ]
)

print(f'Num train samples: {len(train_data)}')
print(f'Num val samples: {len(val_data)}')
print(f'Num test samples: {len(test_data)}')


# ---- MODEL ---------------------------------------------------------------------------

# Load a transformers model
# We use sentence-transformers/paraphrase-MiniLM-L6-v2
# https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L6-v2

TRANSFORMER_MODEL = 'sentence-transformers/paraphrase-MiniLM-L6-v2'
MAX_SEQ_LEN = 50

tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL)


def collate_fn(inputs: List[str]):
    return tokenizer(
        inputs,
        truncation=True,
        max_length=MAX_SEQ_LEN,
        padding=True,
        return_tensors='pt',
    )


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


class TransformerEmbedder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained(TRANSFORMER_MODEL)

    def forward(self, inputs):
        out = self.model(**inputs)
        return mean_pooling(out, inputs['attention_mask'])


# ---- FINE-TUNING ---------------------------------------------------------------------


EPOCHS = 6
BATCH_SIZE = 256
LEARNING_RATE = 1e-4
NUM_WORKERS = 8
NUM_ITEMS_PER_CLASS = 4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def configure_optimizer(model: torch.nn.Module):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=50,
        num_training_steps=EPOCHS * math.ceil(len(train_data) / BATCH_SIZE),
    )
    return optimizer, scheduler


# Let's now run the fine-tuning!
evaluation_callback = EvaluationCallback(val_data, limit=20, num_workers=NUM_WORKERS)
wandb_logger = WandBLogger()
early_stopping = EarlyStopping(patience=1, monitor='average_precision')
best_model_ckpt = BestModelCheckpoint(
    save_dir='checkpoints', monitor='average_precision'
)

finetuned_model = finetuner.fit(
    TransformerEmbedder(),
    train_data=train_data,
    eval_data=val_data,
    loss=TripletLoss(
        distance='cosine',
        margin=0.4,
        miner=TripletEasyHardMiner(pos_strategy='easy', neg_strategy='hard'),
    ),
    configure_optimizer=configure_optimizer,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    num_items_per_class=NUM_ITEMS_PER_CLASS,
    device=DEVICE,
    collate_fn=collate_fn,
    callbacks=[evaluation_callback, wandb_logger, early_stopping, best_model_ckpt],
)

# Now we will evaluate both pre-trained and fine-tuned models in our test data

pretrained_model = TransformerEmbedder()
evaluator = Evaluator(test_data, embed_model=pretrained_model)
pretrained_metrics = evaluator.evaluate(
    limit=30,
    num_workers=NUM_WORKERS,
    batch_size=BATCH_SIZE,
    device=DEVICE,
    collate_fn=collate_fn,
)
print('Evaluating PRE-TRAINED model on test data:')
print('\n'.join([f'{k}:{v:.3f}' for k, v in pretrained_metrics.items()]))

evaluator = Evaluator(test_data, embed_model=finetuned_model)
finetuned_metrics = evaluator.evaluate(
    limit=30,
    num_workers=NUM_WORKERS,
    batch_size=BATCH_SIZE,
    device=DEVICE,
    collate_fn=collate_fn,
)
print('Evaluating FINE-TUNED model on test data:')
print('\n'.join([f'{k}:{v:.3f}' for k, v in finetuned_metrics.items()]))


# ---- INFERENCE -----------------------------------------------------------------------


def predict_intents(
    utterance: str,
    model: torch.nn.Module,
    index: DocumentArray,
    k: int = 20,
) -> List[Tuple[str, float]]:
    """
    Find top k nearest neighbors in a search query
    and compute intents by aggregating distances
    """
    doc = Document(text=utterance)
    embed(
        DocumentArray(doc),
        embed_model=model,
        device=DEVICE,
        batch_size=1,
        collate_fn=collate_fn,
    )
    doc.match(index, limit=k)

    intents = [m.tags['finetuner_label'] for m in doc.matches]
    distances = [m.scores['cosine'].value for m in doc.matches]
    sum_distances = sum(distances)
    scores = [dist / sum_distances for dist in distances]

    output = defaultdict(float)
    for intent, score in zip(intents, scores):
        output[intent] += score

    return sorted(list(output.items()), key=lambda x: x[1], reverse=True)

# The method above allows us to compute intents on search queries
# using our embedding model
# Let's try it out!


pretrained_model = TransformerEmbedder()

# First let's index our data
pretrained_index = deepcopy(train_data)
finetuned_index = deepcopy(train_data)

embed(
    pretrained_index,
    embed_model=pretrained_model,
    device=DEVICE,
    batch_size=BATCH_SIZE,
    collate_fn=collate_fn,
)
embed(
    finetuned_index,
    embed_model=finetuned_model,
    device=DEVICE,
    batch_size=BATCH_SIZE,
    collate_fn=collate_fn,
)

# Let's predict!
utterance = 'Where do you think I should travel to this Christmas?'
intents_pretrained = predict_intents(utterance, pretrained_model, pretrained_index, k=1)
intents_finetuned = predict_intents(utterance, finetuned_model, finetuned_index, k=1)
print(f'Utterance: {utterance}')
print(f'Predicted intents (pre-trained model, k=1): {intents_pretrained}')
print(f'Predicted intents (fine-tuned model, k=1): {intents_finetuned}')

# Let's try with k=20
intents_pretrained = predict_intents(
    utterance, pretrained_model, pretrained_index, k=20
)
intents_finetuned = predict_intents(utterance, finetuned_model, finetuned_index, k=20)
print(f'Utterance: {utterance}')
print(f'Predicted intents (pre-trained model, k=20): {intents_pretrained}')
print(f'Predicted intents (fine-tuned model, k=20): {intents_finetuned}')

# How about utterances with two intents?
utterance = 'What is my location right now? Can you share it with Dave?'
intents_pretrained = predict_intents(
    utterance, pretrained_model, pretrained_index, k=20
)
intents_finetuned = predict_intents(utterance, finetuned_model, finetuned_index, k=20)
print(f'Utterance: {utterance}')
print(f'Predicted intents (pre-trained model, k=20): {intents_pretrained}')
print(f'Predicted intents (fine-tuned model, k=20): {intents_finetuned}')

# Since we have a way to predict classes in text data, we can evaluate the model
# using classification metrics. Let's try to compare pre-trained and fine-tuned
# models in terms of accuracy

true_intents = [doc.tags['finetuner_label'] for doc in test_data]
pretrained_predicted_intents = [
    predict_intents(doc.text, pretrained_model, pretrained_index, k=20)[0][0]
    for doc in test_data
]
finetuned_predicted_intents = [
    predict_intents(doc.text, finetuned_model, finetuned_index, k=20)[0][0]
    for doc in test_data
]

pretrained_acc = sum(
    [int(t == p) for t, p in zip(true_intents, pretrained_predicted_intents)]
) / len(test_data)

finetuned_acc = sum(
    [int(t == p) for t, p in zip(true_intents, finetuned_predicted_intents)]
) / len(test_data)

print(f'Pre-trained model accuracy: {pretrained_acc:.3f}')
print(f'Fine-tuned model accuracy: {finetuned_acc:.3f}')

```
````
