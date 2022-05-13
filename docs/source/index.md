# Welcome to Finetuner!

```{include} ../README.md
:start-after: <!-- start elevator-pitch -->
:end-before: <!-- end elevator-pitch -->
```

## Why I need it?

Search quality matters.
When you bring pre-trained model to encode your data as embeddings.
You are likely to get irrelevant search results.
Deep learning models are mostly likely to be trained on a large dataset.
While this large dataset always has a different *data distribution* over your own dataset.
We refer it as *distribution shift*.

**Finetuner** aims at take a pre-trained model from a large dataset,
fine-tune the parameters of the model on your dataset.
Once fine-tune is done,
you get a model adapted to your domain.
This new model leverage better search performance on your-task-of-interest.

Fine-tuning is non-trivial for business owners/engineers who lack of practical deep learning knowledge.
**Finetuner** could be as easy as:

1. Login to Jina ecosystem with `finetuner.login`.
2. Specify your `DocumentArray` as input.
3. Specify one of the model backbones we supported.
4. Call the `finetuner.fit` function and everything happens in the Jina cloud.
5. Call the `finetuner.download` function to get your tuned model.

```{Important}
Not sure which model to use?

Don't worry, call `finetuner.list_models`, we will help you to choose the best fit.
```

## How it works?

**Finetuner** brings SOTA research ideas such as [transfer learning]() and [metric learning]() into production.

+ *Transfer Learning* means we adopt a pre-trained model, while we only re-train part of the deep learning model on our own dataset. This allows fine-tuning much more computational effective if you do not have enough labeled data.
+ *Metric Learning* means **Finetuner** samples training data from your **DocumentArray** as triplets such as `(anchor, positive, negative)`. The objective of fine-tuning is to bring `anchor` as close as `positive` item, while pull `anchor` apart from `negative` item.





## Install

![PyPI](https://img.shields.io/pypi/v/finetuner?color=%23ffffff&label=%20) is the latest version.

Make sure you have Python 3.7+ installed on Linux/Mac/Windows:

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
