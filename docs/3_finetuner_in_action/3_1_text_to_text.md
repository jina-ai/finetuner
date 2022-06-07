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

We will use the Quora Question Pairs dataset to show-case Finetuner for text-to-text search. It consists of a `train` and `test` dataset. Download these as follows:

```
curl -o quora-IR-dataset.zip "https://sbert.net/datasets/quora-IR-dataset.zip"
```