(class_dataset)=
# Class Dataset

A {class}`~fintuner.tuner.dataset.ClassDataset` is a `DocumentArray`, in which each `Document` has a discrete label stored under `.tags['finetuner_label']`, e.g. `red/black/white/yellow`, `shirt/tops/hats`.

Note, the label is not necessarily related to classification - rather, it is there to denote similar `Document`s. That is, all `Document`s with the same label are considered similar, and all `Document`s with different labels are considered dis-similar.

Comparing to {ref}`session-dataset`, here the Document has a flat structure - no `matches` are needed. This is very convenient for those datasets have explicit/implicit labels, no conversion or heavy processing is required.  

## Batch building

A `ClassDataset` works with {class}`~fintuner.tuner.dataset.ClassSampler`. 

The "class" label is used to construct batches in a two-step  procedure. First, a number of different classes will be randomly sampled, and then from each class a number of `Document`s will be sampled to fill the batch.

Specifically, the size of the batch is controlled by the `batch_size` argument, The number of instances per class is computed dynamically, so that the batch is full. The image below illustrates this for `batch_size=8` and `num_items_per_class=2`.

```{figure} ../class-dataset.png
:align: center
:width: 80%
```

## Examples

### Toy example
Here is an example of a toy class dataset

```python
import random

from docarray import DocumentArray, Document

from finetuner.tuner.dataset import ClassDataset, ClassSampler

contents = ['shirt'] * 2 + ['shoe'] * 6 + ['pants'] * 4
contents = [random.sample(['green', 'red', 'yellow'], k=1)[0] + ' ' + c for c in contents]
labels = [1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3]
ds = DocumentArray(
    [Document(content=c, tags={'finetuner_label': l}) for c, l in zip(contents, labels)]
)

cds = ClassDataset(ds)
for b in ClassSampler(cds.labels, batch_size=2):
    print([cds[bb] for bb in b])
```

```text
[('red shirt', 0), ('red shoe', 1)]
[('yellow shoe', 2), ('red shoe', 1)]
[('yellow pants', 2), ('red shirt', 0)]
[('yellow shoe', 1), ('red pants', 2)]
[('yellow shoe', 1), ('green pants', 2)]
[('red pants', 2), ('yellow shoe', 1)]
```

We got 6 batches. One can see that the sampler tries its best effort to pick an item from each class and form a batch.

(build-mnist-data)=
### Fashion-MNIST data

Fashion-MNIST contains 60,000 training images and 10,000 images in 10 classes. Each image is a single channel 28x28 grayscale image.


```{figure} ../fashion-mnist-sprite.png
:align: center
:width: 60%
```

This dataset is by nature a class dataset - each `Document` has a class label (corresponding to one of the 10 classes). The `Document`s contain the following relevant information:

- `.blob`: the numpy array of the image
- `.tags['finetuner_label']`: the class label

One can use {meth}`~finetuner.toydata.generate_fashion` to generate it.
