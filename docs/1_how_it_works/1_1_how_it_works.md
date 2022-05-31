# How Finetuner Works

## Contrastive Metric Learning 👓

From an algorithmic perspective,
**Finetuner** leverage contrastive metric learning approach to improve your model.
How does it work?

### Step 1: Convert your model into an Embedding Model

Finetuner interprets your model architecture,
removes the default *head*, applies *pooling* and freezes layers that do not need to be trained..
For example, if your pre-trained model is trained for cat-dog classification,
Finetuner is going to remove the cat-dog classifier and turn your model into an *embedding model*.

This embedding model does not make predictions or outputs a probability,
but instead outputs a feature vector to represent your data

### Step 2: Triplets Construction and Training on-the-fly

Finetuner "looks" into the label of your training data.
Each `Document` and `Document`s share the same ****`finetuner_label` is considered as a *Positive*.
In the meanwhile, each `Document` and `Documents` which share a different `finetuner_label` is considered as a *Negative*.

During model fine-tuning, Finetuner is creating *Triplets*  ``(anchor, positive, negative)`` on-the-fly.
Finetuner then uses the triplets to perform training,
the objective is to pull `Document`s belongs to the same class together,
while push the `Document`s belongs to the different class away from each other.



## Cloud-based Fine-tuning ⛅

From engineering perspective,
we have hide all the complexity of machine learning algorithms,
resources setup (such as GPU),
all you need to do is to decide your backbone model and prepare your training data.

Once you logged into Jina Ecosystem with `finetuner.login()`,
Finetuner will push your training data into our *Cloud Artifact Storage* (only visible to you).
At the same time, we will spin-up an isolated computational resource
with proper memory, CPU, GPU dedicated for your fine-tune job.

Once fine-tuning is done, Finetuner will again push your `tuned_model` to the *Cloud Artifact Storage*
and available for you to pull it back to your machine,
that's it!

On the other hand,
if you have certain level of machine learning knowledge,
Finetuner gives you enough flexibility to adjust the training parameters.
This will be explained in the later section.
