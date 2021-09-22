# Overview

```{caution}
This section is not meant to be public. It is a collection of my thoughts on the high-level design & positioning of the Finetuner project in the Jina AI landscape. Please do not share any content below.

Some of the paragraph may be published later after my revision.
```
```{tip}
This chapter gives an overview about the Finetuner design. To understand how do I reach here, please read {ref}`design-philo` and {ref}`design-decision`.
```


By allowing anyone to easily tune any embedding model in an interactive manner, Finetuner solves the **last mile delivery** when developing a neural search app.


## Minimum flexibility, but managed experience

Unlike Jina that aims at flexibility, Finetuner provides users a **managed**, **finetuning** experience **inside Jina ecosystem**.  

```{tip}
What this implies is:
- Developers of Finetuner should not argue about flexibility, it is not part of the design;
- Finetuner is not a general training framework for training model from scratch;
- Finetuner must share the same data interface with Jina core;
```

## One-liner interface

Finetuner has a single high-level interface `finetuner.fit()`, one can use it via:

```python
import finetuner as ft

ft.fit(...)
```

## Framework-agnostic

Finetuner supports Keras, Pytorch and Paddle as the deep learning backend; with the same look, feel and behave on the high-level API. Users can stick to their most comfortable framework and they will enjoy a consistent experience when using Finetuner.  

(three-pillars)=
## Three pillars

Finetuner project is composed of three components:
- **Tuner**: to tune any embedding model for better embedding on labeled data;
- **Tailor**: to trim any deep neural network into an embedding model;
- **Labeler**: a UI for interactive labeling and conduct [active learning](https://en.wikipedia.org/wiki/Active_learning_(machine_learning)) via Tuner.


