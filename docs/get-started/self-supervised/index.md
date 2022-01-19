# Finetuning ResNet50 on Totally Looks Like Dataset with Self-Supervised Pre-training

In a real-world scenario, not every dataset has labels while fine-tuning.
And it is non-trivial to collect a large number of quality labels on customized data.
Luckily, Finetuner allows you to Finetune a pre-trained {term}`embedding_model` without full supervision.

This tutorial uses totally looks like dataset, but we'll pre-train an {term}`embedding model` without using labels.
This is usually referred to as **self-supervised pre-training**.
After this step, we utilize 10% of the dataset with labels to improve the search quality.

This tutorial is a follow-up of [Finetuning ResNet50 on Totally Looks Like Dataset](../totally-looks-like/index.md).
For dataset introduction or fully supervised model fine-tuning,
please refer to that tutorial.

## Preparing Training data

The [Totally Looks Like dataset](https://sites.google.com/view/totally-looks-like-dataset) (TLL) dataset consists of 6016 pairs of images (12032 in total).
We will download `left.zip` and `right.zip`,
each of them consists of 6016 images which can be formed into pairs based on the same file name.

```shell
pip install gdown
pip install finetuner
pip install torchvision

gdown https://drive.google.com/uc?id=1jvkbTr_giSP3Ru8OwGNCg6B4PvVbcO34
gdown https://drive.google.com/uc?id=1EzBZUb_mh_Dp_FKD0P4XiYYSd0QBH5zW

unzip left.zip
unzip right.zip
```

Afterward, we load all images from unzipped `left` and `right` folders and turn them into sorted order as jina `DocumentArray`.
70% of the dataset will be used to self-supervised pre-training.
10% of the dataset will be used to model fine-tuning (on self-supervised embedding model).
While 20% of the dataset will be used to evaluate the quality of embeddings on the search task.

```python
from docarray import DocumentArray

left_da = DocumentArray.from_files('left/*.jpg')
right_da = DocumentArray.from_files('right/*.jpg')

left_da = DocumentArray(sorted(left_da, key=lambda x: x.uri))
right_da = DocumentArray(sorted(right_da, key=lambda x: x.uri))

# we use 70% of data for self-supervised pre-training(no labels).
ratio = 0.7
train_size = int(ratio * len(left_da))

train_da = left_da[:train_size] + right_da[:train_size]
```

## Self-Supervised Pre-training

## Fine-tuning Given Limited Supervision

## Result Evaluation

## Wrapping Up