# Class Dataset Format

A class dataset is a `DocumentArray`, in which each `Document` has a discrete label stored under `.tags['finetuner_label']`, e.g. `red/black/white/yellow`, `shirt/tops/hats`.

The label is not necessarily related to classification - rather, it is there to denote similar `Document`s. That is, all `Document`s with the same label are considered similar, and all `Document`s with different labels are considered dis-similar.

Comparing to {term}`session dataset`, the Document has a flat structure - no `matches` are needed. This is very convenient for those datasets have explicit/implicit labels, no conversion or heavy processing is required.  

## Training-time behavior

During training, this "class" label is used to construct batches in a two-step sampling procedure. In each batch, a number of different classes will be randomly sampled, and from each class a number of `Document`s will be sampled to fill the batch.

Specifically, the size of the batch is controlled by the `batch_size` argument, The number of instances per class is computed dynamically, so that the batch is full. The image below illustrates this for `batch_size=8` and `num_items_per_class=2`.

```{figure} ../class-dataset.png
:align: center
:width: 80%
```

## Examples

Here is an example of a toy class dataset

```python
from jina import DocumentArray, Document

from finetuner.tuner.dataset import ClassDataset
from finetuner.tuner.dataset.samplers import RandomClassBatchSampler

labels = [1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3]
ds1 = ClassDataset(
    DocumentArray(
        [Document(id=str(idx), tags={'finetuner_label': idx}) for idx in labels]
    )
)

for b in RandomClassBatchSampler(ds1.labels, 2):
    print(b)
```

```text
[8, 4]
[6, 0]
[9, 3]
[1, 5]
[7, 2]
```

We got 5 batches. One can see that the sampler tries its best effort to pick an item from each class and form a batch.

(build-mnist-data)=
Fashion-MNIST contains 60,000 training images and 10,000 images in 10 classes. Each image is a single channel 28x28 grayscale image.


```{figure} ../fashion-mnist-sprite.png
:align: center
:width: 60%
```

This dataset is by nature a class dataset - each `Document` has a class label (corresponding to one of the 10 classes). The `Document`s contain the following relevant information:

- `.blob`: the numpy array of the image
- `.tags['finetuner_label']`: the class label

One can use {meth}`~finetuner.toydata.generate_fashion` to generate it.
