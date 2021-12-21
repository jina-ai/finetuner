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
siamese/triplet network, interactive labeling, layer pruning, weights freezing, dimensionality reduction.

⚛️ **Framework-agnostic**: promise an identical API & user experience on PyTorch, Tensorflow/Keras and PaddlePaddle deep learning backends.

🧈 **Jina integration**: buttery smooth integration with Jina, reducing the cost of context-switch between experiment
and production.

<!-- end elevator-pitch -->

## How does it work

<img src="https://github.com/jina-ai/finetuner/blob/main/docs/img/finetuner-journey.svg?raw=true" alt="Python 3.7 3.8 3.9" title="Finetuner supports Python 3.7 and above">


## Install

Requires Python 3.7+ and *one of* [PyTorch](https://pytorch.org/)(>=1.9) or [Tensorflow](https://tensorflow.org/)(>=2.5) or [PaddlePaddle](https://github.com/PaddlePaddle/Paddle) installed on Linux/MacOS.

```bash
pip install finetuner
```

## [Documentation](https://finetuner.jina.ai)

## Usage

<table>
<thead>
  <tr>
    <th colspan="2" rowspan="2">Usage</th>
    <th colspan="2">Do you have an <a href="https://finetuner.jina.ai/basics/glossary/#term-Embedding-model">embedding model</a>?</th>
  </tr>
  <tr>
    <th>Yes</th>
    <th>No</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td rowspan="2"><b>Do you have <a href="https://finetuner.jina.ai/basics/glossary/#term-Labeled-dataset">labeled data</a>?</b></td>
    <td><b>Yes</b></td>
    <td align="center">🟠</td>
    <td align="center">🟡</td>
  </tr>
  <tr>
    <td><b>No</b></td>
    <td align="center">🟢</td>
    <td align="center">🔵</td>
  </tr>
</tbody>
</table>

### 🟠 Have embedding model and labeled data

Perfect! Now `embed_model` and `labeled_data` are given by you already, simply do:

```python
import finetuner

tuned_model, _ = finetuner.fit(
    embed_model,
    train_data=labeled_data
)
```

### 🟢 Have embedding model and unlabeled data

You have an `embed_model` to use, but no labeled data for finetuning this model. No worry, that's good enough already!
You can use Finetuner to interactive label data and train `embed_model` as below:

```python
import finetuner

tuned_model, _ = finetuner.fit(
    embed_model,
    train_data=unlabeled_data,
    interactive=True
)
```

### 🟡 Have general model and labeled data

You have a `general_model` which does not output embeddings. Luckily you provide some `labeled_data` for training. No
worries, Finetuner can convert your model into an embedding model and train it via:

```python
import finetuner

tuned_model, _ = finetuner.fit(
    general_model,
    train_data=labeled_data,
    to_embedding_model=True,
    output_dim=100
)
```

### 🔵 Have general model and unlabeled data

You have a `general_model` which is not for embeddings. Meanwhile, you don't have labeled data for training. But no
worries, Finetuner can help you train an embedding model with interactive labeling on-the-fly:

```python
import finetuner

tuned_model, _ = finetuner.fit(
    general_model,
    train_data=unlabeled_data,
    interactive=True,
    to_embedding_model=True,
    output_dim=100
)
```

## Finetuning ResNet50 on CelebA

> ⚡ To get the best experience, you will need a GPU-machine for this example. For CPU users, we provide [finetuning a MLP on FashionMNIST](https://finetuner.jina.ai/get-started/fashion-mnist/) and [finetuning a Bi-LSTM on CovidQA](https://finetuner.jina.ai/get-started/covid-qa/) that run out the box on low-profile machines. Check out more examples in [our docs](https://finetuner.jina.ai)!


1. Download [CelebA-small dataset (7.7MB)](https://static.jina.ai/celeba/celeba-img.zip) and decompress it to `'./img_align_celeba'`. [Full dataset can be found here.](https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8?resourcekey=0-5BR16BdXnb8hVj6CNHKzLg)
2. Finetuner accepts Jina `DocumentArray`/`DocumentArrayMemmap`, so we load CelebA image into this format using a generator:
    ```python
    from jina.types.document.generators import from_files

    # please change the file path to your data path
    data = list(from_files('img_align_celeba/*.jpg', size=100, to_dataturi=True))

    for doc in data:
        doc.load_uri_to_image_blob(
            height=224, width=224
        ).set_image_blob_normalization().set_image_blob_channel_axis(
            -1, 0
        )  # No need for changing channel axes line if you are using tf/keras
    ```
3. Load pretrained ResNet50 using PyTorch/Keras/Paddle:
    - PyTorch
      ```python
      import torchvision
      model = torchvision.models.resnet50(pretrained=True)
      ```
    - Keras
      ```python
      import tensorflow as tf
      model = tf.keras.applications.resnet50.ResNet50(weights='imagenet')
      ```
    - Paddle
      ```python
      import paddle
      model = paddle.vision.models.resnet50(pretrained=True)
      ```
4. Start the Finetuner:
    ```python
    import finetuner
    
    finetuner.fit(
        model=model,
        interactive=True,
        train_data=data,
        freeze=True,
        to_embedding_model=True,
        input_size=(3, 224, 224),
        output_dim=100
    )
    ```
5. After downloading the model and loading the data (takes ~20s depending on your network/CPU/GPU), your browser will open the Labeler UI as below. You can now label the relevance of celebrity faces via mouse/keyboard. The ResNet50 model will get finetuned and improved as you are labeling. If you are running this example on a CPU machine, it may take up to 20 seconds for each labeling round.

![Finetuning ResNet50 on CelebA with interactive labeling](docs/get-started/celeba-labeler.gif)


<!-- start support-pitch -->
## Support

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
