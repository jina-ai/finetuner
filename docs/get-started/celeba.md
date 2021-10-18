# Finetuning Pre-Trained ResNet on CelebA Dataset

In this example, we want to "tune" the pre-trained [ResNet](https://arxiv.org/abs/1512.03385) on [CelebA dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), the ResNet model has pre-trained weights on ImageNet.

Precisely, "tuning" means: 
- we set up a Jina search pipeline and will look at the top-K visually similar result;
- we accept or reject the results based on their quality;
- we let the model to remember our feedback and produces better search result.

Hopefully the procedure converges after several rounds; and we get a tuned embedding for better celebrity face search.

## Build embedding model

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

## Prepare data

Now prepare CelebA data for the Finetuner. Note that Finetuner accepts Jina `DocumentArray`/`DocumentArrayMemmap`, so we first convert them into this format.


## Put together

Finally, let's feed the model and the data into the Finetuner:

```python
rv = fit(
    model=model,
    interactive=True,
    train_data=train_generator,
    freeze=True,
    input_size=(3, 224, 224),
    output_dim=512,  # Chop-off the last fc layer and add a trainable Linear layer.
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