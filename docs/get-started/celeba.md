# Finetuning Pre-Trained ResNet on CelebA Dataset

In this example, we want to "tune" the pre-trained [ResNet](https://arxiv.org/abs/1512.03385) on [CelebA dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). Note that, the original weights of the ResNet model was trained on ImageNet.

The Finetuner will work in the following steps: 
- first, we spawn the Labeler that helps us to inspect the top-K visually similar celebrities face images from original ResNet;
- then, with the Labeler UI we accept or reject the results based on their similarities;
- finally, the results are collected at the backend by the Tuner, which "tunes" the ResNet and produces better search result.

Hopefully the procedure converges after several rounds; and we get a tuned embedding for better celebrity face search.

## Prepare CelebA data

Let's first make sure you have downloaded all the images [`img_align_celeba.zip`](https://drive.google.com/file/d/0B7EVK8r0v71pZjFTYXZWM3FlRnM/view?usp=sharing&resourcekey=0-dYn9z10tMJOBAkviAcfdyQ) and [`IdentityCelebA.txt`](https://drive.google.com/file/d/1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS/view?usp=sharing) locally.

```{caution}

Beware that the original CelebA dataset is 1.3GB. In this example, we do not need the full dataset. Here is a smaller version which contains 1000 images from the original dataset. You can [download it from here]().
```

Note that Finetuner accepts Jina `DocumentArray`/`DocumentArrayMemmap`, so we first convert CelebA data into this format.



Since each celebrity has multiple facial images, we first create a `defaultdict` and group these images by their identity:

```python
from collections import defaultdict

DATA_PATH = '~/[YOUR-DIRECTORY]/img_align_celeba/'
IDENTITY_PATH = '~/[YOUR-DIRECTORY]/identity_CelebA.txt'


def group_imgs_by_identity():
    grouped = defaultdict(list)
    with open(IDENTITY_PATH, 'r') as f:
        for line in f:
            img_file_name, identity = line.split()
            grouped[identity].append(img_file_name)
    return grouped
```

Then we create a data generator that yields every image as a `Document` object:

```python
from jina import Document

def train_generator():
    for identity, imgs in group_imgs_by_identity().items():
        for img in imgs:
            d = Document(uri=DATA_PATH + img)
            d.convert_image_uri_to_blob(color_axis=0)
            d.convert_uri_to_datauri()
            yield d
```

## Download the pretrained model

Let's import pre-trained ResNet as our {ref}`embedding model<embedding-model>` using any of the following frameworks.

````{tab} PyTorch

```python
import torchvision

model = torchvision.models.resnet50(pretrained=True)
```

````
````{tab} Keras
```python
import tensorflow as tf

model = tf.keras.applications.resnet50.ResNet50(weights='imagenet')
```
````
````{tab} Paddle
```python
import paddle

model = paddle.vision.models.resnet50(pretrained=True)
```
````




## Put together

Finally, let's feed the model and the data into the Finetuner:

```python
rv = fit(
    model=model,
    interactive=True,
    train_data=train_generator,
    freeze=True,
    input_size=(3, 224, 224),
    output_dim=512,  # Chop-off the last fc layer and add a trainable linear layer.
)
```

## Label interactively

You can now label the data by mouse/keyboard. The model will get trained and improved as you are labeling.

From the backend you will see model's training procedure:

```bash
           Flow@22900[I]:🎉 Flow is ready to use!
	🔗 Protocol: 		HTTP
	🏠 Local access:	0.0.0.0:52621
	🔒 Private network:	172.18.1.109:52621
	🌐 Public address:	94.135.231.132:52621
	💬 Swagger UI:		http://localhost:52621/docs
	📚 Redoc:		http://localhost:52621/redoc
           JINA@22900[I]:Finetuner is available at http://localhost:52621/finetuner
```