# Finetuning Pre-Trained ResNet on CelebA Dataset

In this example, we want to "tune" the pre-trained [ResNet](https://arxiv.org/abs/1512.03385) on [CelebA dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), the ResNet model has pre-trained weights on ImageNet.

Precisely, "tuning" means: 
- we set up a Jina search pipeline and will look at the top-K visually similar result;
- we accept or reject the results based on their quality;
- we let the model to remember our feedback and produces better search result.

Hopefully the procedure converges after several rounds; and we get a tuned embedding for better celebrity face search.

## Build embedding model

Let's import pre-trained ResNet as our {ref}`embedding model<embedding-model>` using any of the following framework.

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