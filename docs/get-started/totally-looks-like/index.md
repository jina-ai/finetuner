# Finetuning ResNet50 on Totally Looks Like Dataset

In this tutorial, we will use Finetuner to finetune embeddings on [Totally Looks Like dataset](https://sites.google.com/view/totally-looks-like-dataset) (TLL).

>Totally-Looks-Like is a dataset and benchmark challenging machine-learned representations to reproduce human perception of image similarity. 

The dataset consist 6016 pairs of images (12032 in total).
And we use Finetuner to tune the model to get a better embedding. But how it works?

Finetuner adopts the idea of `transfer learning` and `metric learning`. The rationale is
1. TLL dataset is a relative small dataset, it's not reasonable to train a large network, such as ResNet on TLL from scratch. So we freeze the weights of a pre-trained ResNet and train a MLP head to adopt our new dataset.
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

left_da.sort(key=lambda x: x.uri)
right_da.sort(key=lambda x: x.uri)

assert len(left_da) == len(right_da) == 6016
assert left_da[0].uri == 'left/00000.jpg'
assert right_da[0].uri == 'right/00000.jpg'

assert left_da[-1].uri == 'left/06015.jpg'
assert right_da[-1].uri == 'right/06015.jpg'
```

## Build Triplets and Transform Training & Test Data

After load data into jina `DocumentArray`, we can prepare triplets for training.
It works as follows:

1. we loop through `left_da` and `right_da` (a pair of similar image).
2. we consider the left image in each pair as `anchor`, right image in each pair as `positive`.
3. for the positive image, we assign the tag `finetuner_label` as 1 to tell Finetuner it's a positive instance.
4. we randomly sample 1 document in `right_da` as a negative image, and we assign `finetuner_label` as `-1` indicates it is a negative instance.
5. we attach `positive` and `negative` instance as `matches` of the `anchor`.

It should be noted that:

1. we must make sure our randomly sample negative image is not itself.
2. in practice, you could sample multiple negative images, such as 1 positive 4 negatives.
3. in practice, you should select hard negatives, for more information, please refer to [losses and miners](https://finetuner.jina.ai/components/tuner/loss/).

```python
from jina import DocumentArray

train = DocumentArray()
test = DocumentArray()

for idx, (anchor, positive) in enumerate(zip(left_da, right_da)):
    positive.tags['finetuner_label'] = 1
    negative = right_da.sample(1)[0] # random sample 1 neg
    assert anchor.uri != negative.uri
    negative.tags['finetuner_label'] = -1
    anchor.matches.extend([positive, negative])
    if idx < 3000:
        train.append(anchor)
    else:
        test.append(anchor)
```

As you might have noticed, for now, we only loaded images from disk, and created Jina `DocumentArray` with a `uri` field.
We need to convert the `uri` to image blobs, so we'll apply some built-in converters on both `train` and `test` set.
The simple "chain of converters" will:

1. read image `uri` into `blob`.
2. normalize the image `blob`.
3. put the `channel axis` from `-1` to `0`, since by default an Image is in `B, H, W, C` format, where `C` indicates the `RGB` channel, we turn it into `B, C, H, W` because it is the expected format for pytorch.

```python
def pre_proc(doc):
    """apply transformation to anchor, pos and neg."""
    for m in doc.matches:
        m.load_uri_to_image_blob().set_image_blob_normalization().set_image_blob_channel_axis(-1, 0)
    return doc.load_uri_to_image_blob().set_image_blob_normalization().set_image_blob_channel_axis(-1, 0)

train.apply(pre_proc)
test.apply(pre_proc)
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
Since the model is pre-trained on ImageNet for a classification task, so the output `fc` layer should not be considered as `embedding layer` .
We can use the pooling layer `adaptiveavgpool2d_173` as output of our embedding model.

## Model Training

Model training 

```python
import finetuner as ft
from finetuner.tuner.pytorch.losses import TripletLoss


m = ft.fit(
    model=resnet,
    train_data=train,
    epochs=10,
    batch_size=128,
    loss=TripletLoss(margin=0.3), 
    learning_rate=1e-5,
    device='cuda',
    to_embedding_model=True,
    input_size=(3, 224, 224),
    layer_name='adaptiveavgpool2d_173',
    freeze=['conv2d_1', 'batchnorm2d_2', 'conv2d_5', 'batchnorm2d_6', 'conv2d_8', 'batchnorm2d_9', 'conv2d_11', 'batchnorm2d_12']
)
```

## Evaluation












