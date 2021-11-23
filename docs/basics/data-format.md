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






(build-qa-data)=
### Covid QA

Covid QA data is a CSV that has 481 rows with the columns `question`, `answer` & `wrong_answer`. 

```{figure} covid-qa-data.png
:align: center
```

To convert this dataset into match data, we build each Document to contain the following relevant information:

- `.text`: the `question` column
- `.matches`: the generated positive & negative matches Document
    - `.text`: the `answer`/`wrong_answer` column
    - `.tags['finetuner_label']`: the match label: `1` or `-1`.

Matches are built with the logic below:

- only allows 1 positive match per Document, it is taken from the `answer` column;
- always include `wrong_answer` column as the negative match. Then sample other documents' answer as negative matches.


```{tip}

The Finetuner codebase contains two synthetic matching data generators for demo and debugging purpose:

- `finetuner.toydata.generate_fashion()`: the generator of Fashion-MNIST matching data.
- `finetuner.toydata.generate_qa()`: the generator of Covid QA matching data.

```
