(data-format)=
# Data Format

Finetuner uses Jina [`Document`](https://docs.jina.ai/fundamentals/document/) as the primitive data type. In
particular, [`DocumentArray`](https://docs.jina.ai/fundamentals/document/documentarray-api/)
and [`DocumentArrayMemap`](https://docs.jina.ai/fundamentals/document/documentarraymemmap-api/) are the input data type
in the high-level `finetuner.fit()` API. This means, your training dataset and evaluation dataset should be stored in `DocumentArray`
or `DocumentArrayMemap`, where each training or evaluation instance is a `Document` object:

```python
import finetuner

finetuner.fit(model,
              train_data=...,
              eval_data=...)
```

This chapter introduces how to construct a `Document` in a way that Finetuner will accept.

There are three different types of datasets:

```{toctree}

datasets/class-dataset
datasets/session-dataset
datasets/unlabeled-dataset
```

Regardless of which dataset format we choose, all documents need a `.content` attribute (which can be `text`, `blob` or `uri` - if you implement [your own loading](#loading-and-preprocessing)).

