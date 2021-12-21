# Finetuning ResNet50 on Totally Looks Like Dataset

In this tutorial, we will use Finetuner to finetune embeddings on [Totally Looks Like dataset](https://sites.google.com/view/totally-looks-like-dataset) (TLL).

>Totally-Looks-Like is a dataset and benchmark challenging machine-learned representations to reproduce human perception of image similarity. 

The dataset consist 6016 pairs of images (12032 in total). 
How it works?

Finetuner adopts the idea of `transfer learning` and `metric learning`. The rationale is
1. TLL dataset is a relative small dataset, it's not reasonable to train a large network, such as ResNet on TLL from scratch. So we freeze the weights of a pre-trained ResNet and only train seval layers of the network.
2. TLL dataset consist pairs of images which can be formed as a positive pair, and a random image can can be considered as a negative pair. We can form a `triplet` and use the Finetuner `TripletLoss`. We expect after fine-tuning, the distance between positive pairs can be pull closer, while the distance between positive and negative pairs can be push away.

## Environment & Data Preparation

We will download `left.zip` and `right.zip`, as stated before,
each of them consist of 6016 images which can be formed into pairs based on the same file name.


```shell
pip install gdown
pip install finetuner
pip install torchvision

gdown https://drive.google.com/uc?id=1jvkbTr_giSP3Ru8OwGNCg6B4PvVbcO34
gdown https://drive.google.com/uc?id=1EzBZUb_mh_Dp_FKD0P4XiYYSd0QBH5zW

unzip left.zip
unzip right.zip
```

In this tutorial, we rely on `torchvision` to get a pre-trained ResNet50 on ImageNet dataset.
We'll need to install jina to prepare training/test data since it is the expected input of Finetuner.
Last but not least, we import finetuner and the loss function `TripletLoss`.

Afterwards, we load all images from unzipped `left` and `right` folder and turn them into sorted order.

```python
from jina import DocumentArray

left_da = DocumentArray.from_files('left/*.jpg')
right_da = DocumentArray.from_files('right/*.jpg')
# we use 80% for training machine learning model.
left_da.sort(key=lambda x: x.uri)
right_da.sort(key=lambda x: x.uri)

ratio = 0.8
train_size = int(ratio * len(left_da))

train_da = left_da[:train_size] + right_da[:train_size]
train_da = train_da.shuffle()
```

## Transform Training Data

After load data into jina `DocumentArray`, we can prepare documents for training.
Finetuner will do the most challenge work for you, all you need to do is to:

1. Assign a label into each `Document` named `finetuner_label` as it's class name.
2. Perform pre-processing for the document. In this case, we load the image from uri, normalize the image and reshape the image from `H, W, C` to `C, H W` will `C` is the color channel of the image.

```python
def assign_label_and_preprocess(doc):
    doc.tags['finetuner_label'] = doc.uri.split('/')[1]
    return doc.load_uri_to_image_blob().set_image_blob_normalization().set_image_blob_channel_axis(-1, 0)

train_da.apply(assign_label_and_preprocess)
```

## Prepare Model and Model Visualization

We create a pre-trained ResNet-50 model from torchvision, and since we want to learn a better `embedding`,
the first thing is to see which layer is suitable for use as `embedding layer`.
You can call `finetuner.display(model, input_size)` to plot the model architecture.

```python
import fintuner as ft
import torchvision

resnet = torchvision.models.resnet50(pretrained=True)
ft.display(resnet, (3, 224, 224))
```

You can get more information in [Tailor docs](https://finetuner.jina.ai/components/tailor/).
Since the model is pre-trained on ImageNet on a classification task, so the output `fc` layer should not be considered as `embedding layer` .
We can use the pooling layer `adaptiveavgpool2d_173` as output of our embedding model.
This layer generates a 2048 dimensional dense embedding as output.

## Model Training

Model training is straitforward in finetuner. 
You'll need to config several hyper-parameters,
plugin your model and training set, that's it.

The script below demonstrates how to combine Tailor + Tuner for model fine-tuning.
The parameter above ``to_embedding_model=True`` are tuner parameters, the rest are tailor parameters.

We save the returned embedding model as ``tuned_model``,
given an input image, at inference time, this model generates a ``representation`` of the image (1024d embeddings).

```python
import finetuner as ft
from finetuner.tuner.pytorch.losses import TripletLoss
from finetuner.tuner.pytorch.miner import TripletEasyHardMiner


tuned_model = ft.fit(
    model=resnet,
    train_data=train_da,
    epochs=6,
    batch_size=128,
    loss=TripletLoss(miner=TripletEasyHardMiner(neg_strategy='hard'), margin=0.3), 
    learning_rate=1e-5,
    device='cuda',
    to_embedding_model=True,
    input_size=(3, 224, 224),
    layer_name='adaptiveavgpool2d_173',
    num_items_per_class=2,
    freeze=['conv2d_1', 'batchnorm2d_2', 'conv2d_5', 'batchnorm2d_6', 'conv2d_8', 'batchnorm2d_9', 'conv2d_11', 'batchnorm2d_12'],
)
```

But how does it work? We'll explain briefly:

1. Finetuner will "look into" your labels defined in the `tag` of the jina `Document`, and find the positive sample and find a hard-negative sample as triplets.
2. Finetuner try to optimize the `TripletLoss` objective, aiming at pull documents with same classes closer, while push documents with different class away.

In research domain, this is normally referred as supervised contrastive metric learning.

## Evaluating the Embedding Quality

We'll use `hit@10` to measure the quality of the representation on search task.
``hit@10`` means for all the test data, how likely the positive `match` ranked within the top 10 matches with respect to the `query` Document we give.

Remind that we have the `train_da` ready, now we need to perform same preprocessing on test da:

```python
def preprocess(doc):
    return doc.load_uri_to_image_blob().set_image_blob_normalization().set_image_blob_channel_axis(-1, 0)

test_left_da = left_da[train_size:]
test_right_da = right_da[train_size:]

test_left_da.apply(preprocess)
test_right_da.apply(preprocess)
```

And we create embeddings on our test set using the fine-tuned model:

```python
# use finetuned model to create embeddings， only test data
test_left_da.embed(tuned_model, device='cuda')
test_right_da.embed(tuned_model, device='cuda')
```

Last but not least, we perform evaluation:

```python
test_left_da.match(test_right_da, limit=10)

def hit_rate(da, topk=1):
    hit = 0
    for d in da:
        for m in d.matches[:topk]:
            if d.uri.split('/')[-1] == m.uri.split('/')[-1]:
                hit += 1
    return hit/len(da)


for k in range(1, 11):
    print(f'hit@{k}:  finetuned: {hit_rate(test_left_da, k):.3f}')
```

And we get:

```console
hit@1:  finetuned: 0.122
hit@2:  finetuned: 0.159
hit@3:  finetuned: 0.184
hit@4:  finetuned: 0.207
hit@5:  finetuned: 0.230
hit@6:  finetuned: 0.251
hit@7:  finetuned: 0.268
hit@8:  finetuned: 0.278
hit@9:  finetuned: 0.294
hit@10:  finetuned: 0.301
```

How much performance gain we got?
We conducted an experiment using pre-trained ResNet50 on ImageNet, by chopping-off the last classification layer as feature extractor.
















