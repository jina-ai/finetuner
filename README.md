<p align="center">
<img src="https://github.com/jina-ai/finetuner/blob/main/docs/_static/finetuner-logo-ani.svg?raw=true" alt="Finetuner logo: Finetuner allows one to finetune any deep Neural Network for better embedding on search tasks. It accompanies Jina to deliver the last mile of performance-tuning for neural search applications." width="150px">
</p>


<p align="center">
<b>Finetuning any deep neural network for better embedding on neural search tasks</b>
</p>

<p align=center>
<a href="https://pypi.org/project/finetuner/"><img src="https://github.com/jina-ai/jina/blob/master/.github/badges/python-badge.svg?raw=true" alt="Python 3.7 3.8 3.9" title="Finetuner supports Python 3.7 and above"></a>
<a href="https://pypi.org/project/finetuner/"><img src="https://img.shields.io/pypi/v/finetuner?color=%23099cec&amp;label=PyPI&amp;logo=pypi&amp;logoColor=white" alt="PyPI"></a>
<a href="https://codecov.io/gh/jina-ai/finetuner"><img src="https://codecov.io/gh/jina-ai/finetuner/branch/main/graph/badge.svg?token=xSs4acAEaJ"/></a>
<a href="https://slack.jina.ai"><img src="https://img.shields.io/badge/Slack-2.2k%2B-blueviolet?logo=slack&amp;logoColor=white"></a>
</p>

<!-- start elevator-pitch -->

Finetuner allows one to tune the weights of any deep neural network for better embeddings on search tasks. It
accompanies [Jina](https://github.com/jina-ai/jina) to deliver the last mile of performance for domain-specific neural search
applications.

🎛 **Designed for finetuning**: a human-in-the-loop deep learning tool for leveling up your pretrained models in domain-specific neural search applications.

🔱 **Powerful yet intuitive**: all you need is `finetuner.fit()` - a one-liner that unlocks rich features such as
siamese/triplet network, metric learning, self-supervised pretraining, layer pruning, weights freezing, dimensionality reduction.

⚛️ **Framework-agnostic**: promise an identical API & user experience on PyTorch, Tensorflow/Keras and PaddlePaddle deep learning backends.

🧈 **DocArray integration**: buttery smooth integration with DocArray, reducing the cost of context-switch between experiment
and production.

<!-- end elevator-pitch -->

## How does it work

<img src="https://github.com/jina-ai/finetuner/blob/main/docs/img/finetuner-journey.svg?raw=true" alt="Python 3.7 3.8 3.9" title="Finetuner supports Python 3.7 and above">


## Install

Requires Python 3.7+ and *one of* [PyTorch](https://pytorch.org/)(>=1.9) or [Tensorflow](https://tensorflow.org/)(>=2.5) or [PaddlePaddle](https://github.com/PaddlePaddle/Paddle) installed on Linux/MacOS.

```bash
pip install finetuner
```

## Finetuning ResNet50 on CelebA

1. Download [CelebA-small dataset (7.7MB)](https://static.jina.ai/celeba/celeba-img.zip) and decompress it to `'./img_align_celeba'`. [Full dataset can be found here.](https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8?resourcekey=0-5BR16BdXnb8hVj6CNHKzLg)
2. Finetuner accepts docarray `DocumentArray`, so we load CelebA image into this format using a generator:
    ```python
    from docarray import DocumentArray
    
    # please change the file path to your data path
    data = DocumentArray.from_files('img_align_celeba/*.jpg')
    
    
    def preproc(doc):
        return (
            doc.load_uri_to_image_tensor(224, 224)
            .set_image_tensor_normalization()
            .set_image_tensor_channel_axis(-1, 0)
        )  # No need for normalization and changing channel axes line if you are using tf/keras
    
    
    data.apply(preproc)
    ```
3. Load pretrained ResNet18 using PyTorch/Keras/Paddle:
    - PyTorch
      ```python
      import torchvision
      resnet = torchvision.models.resnet50(pretrained=True)
      ```
    - Keras
      ```python
      import tensorflow as tf
      resnet = tf.keras.applications.resnet50.ResNet50(weights='imagenet')
      ```
    - Paddle
      ```python
      import paddle
      resnet = paddle.vision.models.resnet50(pretrained=True)
      ```
4. Start the Finetuner:
    ```python
    import finetuner as ft
    
    tuned_model = ft.fit(
        model=resnet,
        train_data=data,
        loss='TripletLoss',
        epochs=20,
        device='cuda',
        batch_size=128,
        to_embedding_model=True,
        input_size=(3, 224, 224), # for keras use (224, 224, 3)
        freeze=False,
    )
    ```


<!-- start support-pitch -->
## Support

- Check out the [Learning Bootcamp](https://learn.jina.ai) to get started with Finetuner.
- Use [Discussions](https://github.com/jina-ai/finetuner/discussions) to talk about your use cases, questions, and
  support queries.
- Join our [Slack community](https://slack.jina.ai) and chat with other Jina community members about ideas.
- Join our [Engineering All Hands](https://youtube.com/playlist?list=PL3UBBWOUVhFYRUa_gpYYKBqEAkO4sxmne) meet-up to discuss your use case and learn Jina's new features.
    - **When?** The second Tuesday of every month
    - **Where?**
      Zoom ([see our public events calendar](https://calendar.google.com/calendar/embed?src=c_1t5ogfp2d45v8fit981j08mcm4%40group.calendar.google.com&ctz=Europe%2FBerlin)/[.ical](https://calendar.google.com/calendar/ical/c_1t5ogfp2d45v8fit981j08mcm4%40group.calendar.google.com/public/basic.ics))
      and [live stream on YouTube](https://youtube.com/c/jina-ai)
- Subscribe to the latest video tutorials on our [YouTube channel](https://youtube.com/c/jina-ai)

## Join Us

Finetuner is backed by [Jina AI](https://jina.ai) and licensed under [Apache-2.0](./LICENSE). [We are actively hiring](https://jobs.jina.ai) AI engineers, solution engineers to build the next neural search ecosystem in opensource.

<!-- end support-pitch -->
