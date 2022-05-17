# Welcome to Finetuner!

```{include} ../README.md
:start-after: <!-- start elevator-pitch -->
:end-before: <!-- end elevator-pitch -->
```

Finetuner is an open-source offering by [Jina AI](https://jina.ai/) ✨

It enables users to fine-tune large pre-trained deep learning models in their specific domains and datasets. It handles the
infrastructure and the complexity of the fine-tuning task and provides a simple interface to submit fine-tuning jobs on the Jina Cloud.

Finetuner primarily targets business users and engineers with limited knowledge in Machine Learning, but also attempts to expose
lots of configuration options for experienced professionals!

## Overview

### Why do I need this? 🤔

Search quality matters. When you bring a pre-trained model to encode your data to embeddings, you are likely to get irrelevant search results.
Pre-trained deep learning models are usually trained on large-scale datasets, that have a different *data distribution* over your own datasets or domains.
This is referred to as a *distribution shift*.

**Finetuner** provides a solution to this problem by leveraging a pre-trained model from a large dataset and fine-tuning the parameters of
this model on your dataset.

Once fine-tuning is done, you get a model adapted to your domain. This new model leverages better search performance on your-task-of-interest.

Fine-tuning a pre-trained model includes a certain complexity and requires Machine Learning plus domain knowledge (on NLP, Computer Vision e.t.c).
Thus, it is a non-trivial task for business owners and engineers who lack the practical deep learning knowledge. **Finetuner** attempts
to address this by providing a simple interface, which can be as easy as:

1. Login to the Jina ecosystem with `finetuner.login()`.
2. Specify your `DocumentArray` as input. 
3. Specify one of the model backbones we support.
4. Call the `finetuner.fit()` function and submit your fine-tuning job in the cloud.
5. Monitor the status and the logs of your job, via `run.status()` and `run.logs()`.
6. Call the `finetuner.download()` function to get your tuned model.

Submitted fine-tuning jobs run efficiently on the Jina Cloud on either CPU or GPU enabled hardware.

Finetuner fully owns the complexity of setting up and maintaining the model training infrastructure plus the complexity of delivering SOTA training
methods to production use cases.

```{Important}
Not sure which model to use?

Don't worry, call `finetuner.list_models()` and we will help you choose the best fit.
```

### How it works? 🧐

**Finetuner** brings SOTA research ideas from [transfer learning](https://en.wikipedia.org/wiki/Transfer_learning), [representation learning](https://en.wikipedia.org/wiki/Transfer_learning) and *metric learning* into production.

+ *Transfer Learning* means we adapt a pre-trained model, while we only re-train part of the deep learning model on our own dataset. This makes fine-tuning more effective in cases where you do not have enough labeled data.
+ *Metric Learning* means **Finetuner** samples training data from your **DocumentArray**s as triplets such as `(anchor, positive, negative)`. The objective of fine-tuning is to bring `anchor`s as close as possible to `positive` items, while pulling `anchor`s apart from `negative` items.


## Installation 🚀

![PyPI](https://img.shields.io/pypi/v/finetuner?color=%23ffffff&label=%20) is the latest version.

Make sure you have `Python 3.7+` installed on Linux/Mac/Windows:

````{tab} Basic install

```bash
pip install finetuner
```

No extra dependency will be installed.
````

````{tab} Basic install via Conda

```bash
conda install -c conda-forge finetuner
```

No extra dependency will be installed.
````

Check your installation with:
```pycon
>>> import finetuner
>>> finetuner.__version__
'0.1.0'
```

```{important}
Jina 3.x users do not need to install `docarray` separately, as it is shipped with Jina. To check your Jina version, type `jina -vf` in the console.

However, if the printed version is smaller than `0.1.0`, say `0.0.x`, then you are 
not installing `docarray` correctly. You are probably still using an old `docarray` shipped with Jina 2.x. 
```

## Getting started

Submitting your job in Jina Cloud in straight-forward. Log-in to Jina Cloud and then call `finetuner.fit()`:

```python
import finetuner

from docarray import DocumentArray

finetuner.login()
train_data = DocumentArray(...)
run = finetuner.fit(train_data=train_data, model='resnet50')
print(run.logs())
```


## Recipes

Add config files as recipes.



```{include} ../README.md
:start-after: <!-- start support-pitch -->
:end-before: <!-- end support-pitch -->
```

```{toctree}
:caption: Get Started
:hidden:

get-started/what-is
```

```{toctree}
:caption: User Guides
:hidden:

fundamentals/document/index
fundamentals/documentarray/index
fundamentals/dataclass/index
datatypes/index
```

```{toctree}
:caption: Integrations
:hidden:

advanced/document-store/index
fundamentals/jina-support/index
fundamentals/notebook-support/index
advanced/torch-support/index
fundamentals/fastapi-support/index
advanced/graphql-support/index
```

```{toctree}
:caption: Developer References
:hidden:
:maxdepth: 1

api/docarray
proto/index
changelog/index
```

---
{ref}`genindex` | {ref}`modindex`
