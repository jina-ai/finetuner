# Finetuning Pretrained ResNet for Celebrity Face Search

```{tip}
For this example, you will need a GPU machine to enable the best experience.
```

In this example, we want to "tune" the pre-trained [ResNet](https://arxiv.org/abs/1512.03385) on [CelebA dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). Note that, the original weights of the ResNet model were trained on ImageNet.

The Finetuner will work in the following steps: 
- first, we spawn the Labeler that helps us to inspect the top-K visually similar celebrities face images from original ResNet;
- then, with the Labeler UI we accept or reject the results based on their similarities;
- finally, the results are collected at the backend by the Tuner, which "tunes" the ResNet and produces better search results.

Hopefully the procedure converges after several rounds; and we get a tuned embedding for better celebrity face search.

## Prepare CelebA data

Let's first make sure we have downloaded all the images [`img_align_celeba.zip`](https://drive.google.com/file/d/0B7EVK8r0v71pZjFTYXZWM3FlRnM/view?usp=sharing&resourcekey=0-dYn9z10tMJOBAkviAcfdyQ) and [`IdentityCelebA.txt`](https://drive.google.com/file/d/1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS/view?usp=sharing) locally.

```{caution}
Beware that the original CelebA dataset is 1.3GB. In this example, we do not need the full dataset. Here is a smaller version which contains 1000 images from the original dataset. You can [download it from here](https://static.jina.ai/celeba/celeba-img.zip).
```

Note that Finetuner accepts Jina `DocumentArray`/`DocumentArrayMemmap`, so we first load CelebA data into this format using a generator:

```python
from docarray.document.generators import from_files

# please change the file path to your data path
data = list(from_files('img_align_celeba/*.jpg', size=100, to_dataturi=True))

for doc in data:
    doc.load_uri_to_image_blob(
        height=224, width=224
    ).set_image_blob_normalization().set_image_blob_channel_axis(
        -1, 0
    )  # No need for changing channel axes line if you are using tf/keras
```

## Load the pretrained model

Let's import a pretrained ResNet50 as our base model. ResNet50 is implemented in PyTorch, Keras and Paddle. You can choose the framework you are the most comfortable with:

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

Finally, let's start the Finetuner. Note that we freeze the weights of the original ResNet and tune only the last linear layer, a procedure that leverages the Tailor component underneath:

```{code-block} python
---
emphasize-lines: 5, 8
---
import finetuner

finetuner.fit(
    model=model,
    interactive=True,
    train_data=data,
    freeze=True,
    to_embedding_model=True,
    input_size=(3, 224, 224),
    freeze=False,
)
```

Note how we specify `interactive=True` and `to_embedding_model=True` in the code above, to activate the Labeler and the Tailor, respectively.

`input_size` is not required when you using Keras as the backend.

## Label interactively

After running the script, the browser will open the Labeler UI. You can now label the data by mouse/keyboard. The model will get trained and improved as you are labeling. If you are running this example on a CPU machine, it can take up to 20 seconds for each labeling round. 

```{figure} celeba-labeler.gif
:align: center
```

On the backend, you should be able to see the training procedure in the terminal.

```console
           Flow@6620[I]:🎉 Flow is ready to use!
	🔗 Protocol: 		HTTP
	🏠 Local access:	0.0.0.0:61622
	🔒 Private network:	172.18.1.109:61622
	🌐 Public address:	94.135.231.132:61622
	💬 Swagger UI:		http://localhost:61622/docs
	📚 Redoc:		http://localhost:61622/redoc
UserWarning: ignored unknown argument: ['thread']. (raised from /Users/hanxiao/Documents/jina/jina/helper.py:685)
⠴ Working... ━╸━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 0:00:00 estimating...            JINA@6620[I]:Finetuner is available at http://localhost:61622/finetuner
⠏ Working... ━━╸━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 0:00:00  0.0 step/s UserWarning: The given NumPy array is not writeable, and PyTorch does not support non-writeable tensors. This means you can write to the underlying (supposedly non-writeable) NumPy array using the tensor. You may want to copy the array to protect its data or make it writeable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at  ../torch/csrc/utils/tensor_numpy.cpp:180.) (raised from /Users/hanxiao/Documents/trainer/finetuner/labeler/executor.py:53)
⠦       DONE ━━━━━━━━━━━╸━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 0:00:06  1.4 step/s 11 steps done in 6 seconds
⠙       DONE ━━╸━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 0:00:03  0.3 step/s T: Loss=    0.75
```