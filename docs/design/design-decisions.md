(design-decision)=
# Decisions

```{caution}
This section is not meant to be public. It is a collection of my thoughts on the high-level design & positioning of the Finetuner project in the Jina AI landscape. Please do not share any content below.

Some of the paragraph may be published later after my revision.
```

## Single API with minimum arguments 

Finetuner exposes only one public API as the entrypoint: `finetuner.fit`. At its minimum form, it requires only two arguments: the model and the data. 

The interface of `Tuner`,  `Tailor` and `Labeler` must comply with this single API:

````{tab} Use Labeler

```{code-block} python
---
emphasize-lines: 5
---
import finetuner as ft

ft.fit(embed_model,
       train_data
       interactive=True)
```
````

````{tab} Use Tuner with Tailor

```{code-block} python
---
emphasize-lines: 3, 5, 6
---
import finetuner as ft

ft.fit(general_model,
       train_data,
       freeze_layer=-1,
       output_dim=128)
```
````

````{tab} Use Tuner

```{code-block} python
---
emphasize-lines: 3
---
import finetuner as ft

ft.fit(embed_model,
       train_data)
```
````

All other arguments are considered as optional and their default values should be carefully chosen by us (i.e. the developer of Finetuner).

Unlike Jina that aims at flexibility, Finetuner provides users a **managed**, **finetuning** experience **inside Jina ecosystem**. For Finetuner, "flexibility" means having more `kwargs` in `finetuner.fit`.   


(embedding-model)=
## Three pillars design


To understand why `Tuner`, `Labeler`, `Tailor` three pillars exist, let me first clarify the two concepts:

- **Embedding model**: a DNN takes an arbitary `ndarray` input (image/text/sequence) with size `[B x ... x ... x ...]` and output a `ndarray` with size `[B x D]`, where `B` is the batch size and the embedded vector is in $R^D$.
- **General model**: a DNN model that does not output `[B x D]`. For example, classification model, objective detection model, next token prediction model, regression model are not embedding model as their output is not a batch of $R^D$ vectors.

Despite the difference on the model architecture, the weights also differs. The weights of a general model is often trained end-to-endly for a specific task. The weights of an embedding model is often initialized from a general model, but then truncated and tuned for a new downstream task.

```{tip}
In my previous slides/talks, I sometimes called the embedding model as bottleneck model, or a model with a bottleneck layer.
```

### Tuner

In a Jina pipeline, the embedding model is the important component for getting the representation. The quality of this representation often directly determines the search quality. On contrary, the general model is nothing interesting to Jina.

That is, tuning the performance of the search equals to tuning the embedding model. {ref}`This is exactly the definition of Tuner<three-pillars>`.

The actual training of Finetuner happens inside the Tuner.

### Tailor 

But where does this embedding model come from? As I said, general models widely exist, and only few of them can be directly used as embedding models. So how can one get a embedding model? There are two ways:
- building from scratch;
- converting a general model to a embedding model.

In [my blog post on the new AI supply chain](https://hanxiao.io/2019/07/29/Generic-Neural-Elastic-Search-From-bert-as-service-and-Go-Way-Beyond/?highlight=body%20%3E%20div.wrap%20%3E%20main%20%3E%20div%20%3E%20article%20%3E%20div.post-content%20%3E%20img:nth-child(26)), I already said less & less people will build new model from scratch. Most people will simply use pretrained or preachitectured models. Hence there is a strong & common requirement of "converting" arbitrary general DNN model into embedding model that Tuner could handle. That's what tailor responsible for.

Given a general model (from your colleague or Pytorch/Keras/Huggingface model zoo), Tailor trims, cuts and does micro-operations on its architecture and outputs an embedding model for the Tuner.

Tailor can be considered as a funnel, which enlarges our model landscape and speeds up the adoption of the Finetuner project. 

### Labeler

I have talked about models and tuning. To conduct training we also need labeled data. That's the job of the labeler: to allow human to interactively label search results. Labeler delivers more than a simple annotation UI, it is also responsible for invoking Tuner to do [active learning](https://en.wikipedia.org/wiki/Active_learning_(machine_learning)), so that the next to-be-labeled data is selected according to the latest tuned embedding model.

### Summary

- **Tuner**: to tune any embedding model for better embedding on labeled data;
- **Tailor**: to trim any deep neural network into an embedding model;
- **Labeler**: a UI for interactive labeling and conduct [active learning](https://en.wikipedia.org/wiki/Active_learning_(machine_learning)) via Tuner.

Let's summarize all user entrypoints here
| User| Entrypoint| Est. Quantity | 
|---|---|---|
| Users that have no labeled data  | Labeler (calls Tuner and Tailor in the back)| Majority |
| Users that have no labeled data but have embedding model  | Labeler (calls Tuner in the back) | Majority |
| Users that have labeled data |  Tuner (calls Tailor in the back) | Minority |
| Users that have labeled data and an embedding model|  Tuner| Minority |

````{tip}
The term `Tuner`, `Tailor`, `Labeler` are partially inspired by the movie "Tinker Tailor Soldier Spy".

```{figure} poster.gif
:align: center
:height: 200px
```
````



## Relationship with Jina

Finetuner is not designed to be a general finetuning framework, it fills the gap of the last mile delivery **for Jina users**.

Finetuner is not an independent project. It has one mandatory upstream dependency: Jina.

In particular, Finetuner is implemented as follows to maximize the compatability with Jina:

- The data exchange format of Tuner & Labeler is `DocumentArray` or `DocumentArrayMemmap`, where each sample data is a `Document`. The rich structure of `Document` gets fully leveraged by Finetuner.
- Labeler leverages Jina `Flow` as the backend to train the model on-the-fly.

## DL backend support

The decision on supporting Pytorch, Keras and Paddle as deep learning backend is fixed. We shall try our best effort to maintain a consistent behavior across these three backends. This will make the Finetuner project framework-agnostic, and serve better our diversified community.

This decision mostly affect the implementation and design of `Tuner` and `Tailor`. In particular, `Tuner` is a good example to show such requirement can be done beautifully. Please read the source code behind Tuner and observe the unified interface there.

On the high-level API, PyTorch and Paddle look and feel extremely similar. This reduces 1/3 our development effort. Paddle is also popular in China. By working with Baidu on promoting Finetuner (scheduled a big press release from them in Nov. 2021), this gives us good exposure to Paddle community.

This decision does not affect the implementation of `Labeler`, as it can be seen as UI + `Flow` + `Tuner` + `Tailor`, where the first two components are framework-irrelevant.
