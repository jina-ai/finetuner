# {octicon}`question` How Does it Work?

From an algorithmic perspective,
Finetuner leverages the contrastive approach to improve models for similarity matching.

## Step 1: Build embedding model

Finetuner interprets the backbone model architecture,
removes the default *head*, applies *pooling* and freezes layers that do not need to be trained.
For an image classification task (e.g. cats and dogs),
Finetuner is going to remove the classification head (cat-dog classifier) and turn your model into an *embedding model*.

This embedding model does not make predictions or outputs a probability,
but instead outputs a feature vector to represent your data.

## Step 2: Tuple/Triplet construction

````{tab} Uni-modal (with label)
Finetuner works on labeled data.
It expects either a CSV file or a {class}`~docarray.array.document.DocumentArray` consisting of {class}`~docarray.document.Document`s where each one contains `finetuner_label` corresponding to the class of a specific training example. After receiving a CSV file, its contents are parsed and a {class}`~docarray.array.document.DocumentArray` is constructed.

During the fine-tuning, Finetuner creates Triplets `(anchor, positive, negative)` on-the-fly.
For each anchor,
which can be any training example,
Finetuner looks for a `Document` with the same `finetuner_label` (positive),
and a `Document` with a different `finetuner_label` (negative).
The objective is to pull `Document`s which belong to the same class together,
while pushing the `Document`s which belong to a different class away from each other.
````
````{tab} Cross-modal (without label)
Finetuner works on unlabeled text image pairs.
You can fine-tune a CLIP like model for text to images search directly without any annotations.
It expects either a CSV file or a {class}`~docarray.array.document.DocumentArray` consisting a list of {class}`~docarray.array.document.Document` contains two chunks: an image chunk and a text chunk.

During fine-tuning, Finetuner leverages text-image pairs and jointly optimizes two models (`CLIPTextEncoder` and `CLIPImageEncoder`) with respect to two classification losses: (1) given a text, find the best matching
image and (2) given an image, find the best matching text. Then it aggregates the two losses into the `CLIPLoss`.
At the end, the output embedding of your data from the `CLIPTextEncoder` is comparable against the `CLIPImageEncoder`.
````
````{tab} Uni-modal (without label, Coming Soon)
Finetuner works on unlabeled texts or images.
While this feature is not opened to the user at the moment.
It expects either a CSV file or a {class}`~docarray.array.document.DocumentArray`. Labels are not required.

Finetuner employs a self-supervised learning approach that applies random augmentation to your data and generates two/multiple Views of your data.
These Views can be considered as positives to each other.
It should be noted that self-supervised approach needs a very large amount of training data.
We have postponed rolling this feature out until we have proven its effectiveness.
````

## Step 3: Tuning in the cloud

From an engineering perspective,
we have hidden all the complexity of machine learning algorithms and resource configuration (such as GPUs).
All you need to do is decide on your backbone model and prepare your training data.

Once you have logged in to the Jina Ecosystem with {meth}`~finetuner.login()`,
Finetuner will push your training data into the *Jina AI Cloud* (only visible to you).
At the same time, we will spin-up an isolated computational resource
with proper memory, CPU, GPU dedicated to your fine-tuning job.

Once fine-tuning is done, Finetuner will again push your fine-tuned model to the *Jina AI Cloud*
and make it available for you to pull it back to your machine.
That's it!

On the other hand,
if you have a certain level of machine learning knowledge,
Finetuner gives you enough flexibility to adjust the training parameters.
This will be explained in a later section.
