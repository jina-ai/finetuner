# How Finetuner works

## Contrastive metric learning 👓

From an algorithmic perspective,
**Finetuner** leverages a contrastive metric learning approach to improve your model.
How does it work?

### Step 1: Convert your model into an Embedding Model

Finetuner interprets your model architecture,
removes the default *head*, applies *pooling* and freezes layers that do not need to be trained..
For example, if your pre-trained model is trained for cat-dog classification,
Finetuner is going to remove the cat-dog classifier and turn your model into an *embedding model*.

This embedding model does not make predictions or outputs a probability,
but instead outputs a feature vector to represent your data.

### Step 2: Triplet construction and training on-the-fly

Finetuner works on labeled data.
It expects a `DocumentArray` consisting of `Document`s where each one contains `finetuner_label` corresponding to a class of a specific training example.

During the fine-tuning, Finetuner creates Triplets `(anchor, positive, negative)` on-the-fly.
For each anchor,
which can be any training example,
Finetuner looks for a `Document` with the same `finetuner_label` (positive),
and a `Document` with a different `finetuner_label` (negative).
The objective is to pull `Document`s which belong to the same class together,
while pushing the `Document`s which belong to a different class away from each other.


## Cloud-based fine-tuning ⛅

From an engineering perspective,
we have hidden all the complexity of machine learning algorithms and resources configuration (such as GPU).
all you need to do is to decide your backbone model and prepare your training data.

Once you logged into Jina Ecosystem with `finetuner.login()`,
Finetuner will push your training data into our *Cloud Artifact Storage* (only visible to you).
At the same time, we will spin-up an isolated computational resource
with proper memory, CPU, GPU dedicated for your fine-tune job.

Once fine-tuning is done, Finetuner will again push your `tuned_model` to the *Cloud Artifact Storage*
and make it available for you to pull it back to your machine,
that's it!

On the other hand,
if you have a certain level of machine learning knowledge,
Finetuner gives you enough flexibility to adjust the training parameters.
This will be explained in the later section.
