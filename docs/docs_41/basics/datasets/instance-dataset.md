(instance_dataset)=
# Instance dataset

{class}`~fintuner.tuner.dataset.InstanceDataset` is similar to {ref}`unlabeled_dataset` - in both dataset you provide a `DocumentArray`, where each `Document` only needs to have content, and does not need any label stored in `tags`.

The difference is that {class}`~fintuner.tuner.dataset.InstanceDataset` considers each element (instance) to be its own class, and gives it a unique label. This is used for self-supervised learning - in particular, it allows {class}`~fintuner.tuner.dataset.InstanceSampler` to put multiple copies of an instance in a batch.

## Batch building

Here's an example demonstrating how batches built with {class}`~fintuner.tuner.dataset.InstanceDataset` and {class}`~fintuner.tuner.dataset.InstanceSampler` look like

```python
from docarray import Document, DocumentArray
from finetuner.tuner.dataset import InstanceDataset, InstanceSampler

data = DocumentArray([
    Document(text='item 1'),
    Document(text='item 2'),
    Document(text='item 3'),
    Document(text='item 4')
])
dataset = InstanceDataset(data)
sampler = InstanceSampler(num_instances=len(dataset), batch_size=4)

for i, batch in enumerate(sampler):
    print(f'Batch {i+1}')
    batch_text = [dataset[ind][0] for ind in batch]
    batch_labels = [dataset[ind][1] for ind in batch]
    print(f'Texts: {batch_text}')
    print(f'Labels: {batch_labels}\n')
```
```console
Batch 1
Texts: ['item 4', 'item 1', 'item 4', 'item 1']
Labels: [3, 0, 3, 0]

Batch 2
Texts: ['item 3', 'item 2', 'item 3', 'item 2']
Labels: [2, 1, 2, 1]
```

As we can see, in each batch every instance was repeated two times. If we had also applied random augmentation using `preprocess_fn`, we would have two different items for each label - and this is the required input for self-supervised training.


