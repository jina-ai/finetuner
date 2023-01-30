# {octicon}`question` How Does it Work?

Finetuner is a framework for using the contrastive learning approach to improve similarity matching with models that encode data into embeddings.
This involves three steps:

## Step 1: Build an embedding model

Finetuner takes an existing, pre-trained model, typically called the __backbone__, and analyzes its architecture.
If this model does not already produce embeddings or the architecture is not suitable for training, Finetuner is able to remove the default *head* (the last layers of the network), add new projection layers, apply *pooling*, and freeze layers that do not need to be trained.

For instance, Finetuner will turn an image classification model, e.g., for separating cats from dogs, into an *embedding model* 
by removing its last layer - the classification head (cat-dog classifier).

This embedding model does not make predictions or output a probability,
but instead outputs a feature vector (an __embedding__) that represents its input.

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
Finetuner works on unlabeled text-image pairs.
You can fine-tune a CLIP-like model for text to images search directly without any labels.
It expects either a CSV file or a {class}`~docarray.array.document.DocumentArray` consisting a list of {class}`~docarray.array.document.Document` that contain two chunks: an image chunk and a text chunk.

During fine-tuning, Finetuner leverages text-image pairs and jointly optimizes two models (`CLIPTextEncoder` and `CLIPImageEncoder`) with respect to two classification losses: (1) given a text, find the best matching
image and (2) given an image, find the best matching text. Then it aggregates the two losses into the `CLIPLoss`.
At the end, the output embedding of your data from the `CLIPTextEncoder` is comparable to `CLIPImageEncoder`.
````

## Step 3: Tuning in the cloud

From an operational perspective,
we have hidden all the complexity of machine learning algorithms and resource configuration (such as GPUs).
All you need to do is decide on your backbone model and prepare your training data.

Once you have logged in to the Jina Ecosystem with {meth}`~finetuner.login()`,
Finetuner will push your training data into the *Jina AI Cloud* (only visible to you).
At the same time, we will spin-up an isolated computational resource
with proper memory, CPU, and a GPU dedicated to your fine-tuning job.

Once fine-tuning is done, Finetuner will push your fine-tuned model to the *Jina AI Cloud*
and make it available for you to download.
That's it!

On the other hand,
if you have a certain level of machine learning knowledge,
Finetuner gives you enough flexibility to adjust the training parameters.
This will be explained in a later section.
