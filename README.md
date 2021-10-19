<p align="center">
<img src="https://github.com/jina-ai/finetuner/blob/main/docs/_static/logo-light.svg?raw=true" alt="Finetuner logo: Finetuner allows one to finetune any deep Neural Network for better embedding on search tasks. It accompanies Jina to deliver the last mile of performance-tuning for neural search applications." width="150px">
</p>


<p align="center">
<b>Finetuning any deep neural network for better embedding on neural search tasks</b>
</p>

<p align=center>
<a href="https://pypi.org/project/finetuner/"><img src="https://github.com/jina-ai/jina/blob/master/.github/badges/python-badge.svg?raw=true" alt="Python 3.7 3.8 3.9" title="Finetuner supports Python 3.7 and above"></a>
<a href="https://pypi.org/project/finetuner/"><img src="https://img.shields.io/pypi/v/finetuner?color=%23099cec&amp;label=PyPI&amp;logo=pypi&amp;logoColor=white" alt="PyPI"></a>
<a href="https://codecov.io/gh/jina-ai/finetuner"><img src="https://codecov.io/gh/jina-ai/finetuner/branch/main/graph/badge.svg?token=xSs4acAEaJ"/></a>
<a href="https://slack.jina.ai"><img src="https://img.shields.io/badge/Slack-1.8k%2B-blueviolet?logo=slack&amp;logoColor=white"></a>
</p>

<!-- start elevator-pitch -->

Finetuner allows one to tune the weights of any deep neural network for better embedding on search tasks. It
accompanies [Jina](https://github.com/jina-ai/jina) to deliver the last mile of performance-tuning for neural search
applications.

🎛 **Designed for finetuning**: a machine learning-powered human-in-the-loop tool for leveling up your pretrained models in neural search applications.

🔱 **Powerful yet intuitive**: all you need is `finetuner.fit()` - a one-liner that unlocks rich features such as
siamese/triplet network, interactive labeling, layer trimming, weights freezing, dimensionality reduction.

⚛️ **Framework-agnostic**: promise an identical API experience on [Pytorch](https://pytorch.org/)
, [Keras](https://keras.io/) or [PaddlePaddle](https://github.com/PaddlePaddle/Paddle) deep learning backends.

🧈 **Jina integration**: buttery smooth integration with Jina, reducing the cost of context-switch between experimenting
and production.

<!-- end elevator-pitch -->

## How does it work

![](docs/img/finetuner-journey.svg)


## Install

Make sure you have Python 3.7+ and one of Pytorch (>=1.9), Tensorflow (>=2.5) or PaddlePaddle installed on Linux/MacOS.

```bash
pip install finetuner
```

## [Documentation](https://finetuner.jina.ai)

## Usage

<table>
<thead>
  <tr>
    <th colspan="2" rowspan="2">🪄 Usage</th>
    <th colspan="2">Do you have an embedding model?</th>
  </tr>
  <tr>
    <th>Yes</th>
    <th>No</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td rowspan="2"><b>Do you have labeled data?</b></td>
    <td><b>Yes</b></td>
    <td align="center">1️⃣</td>
    <td align="center">3️⃣</td>
  </tr>
  <tr>
    <td><b>No</b></td>
    <td align="center">2️⃣</td>
    <td align="center">4️⃣</td>
  </tr>
</tbody>
</table>

### 1️⃣ Have embedding model and labeled data

Perfect! Now `embed_model` and `labeled_data` are given by you already, simply do:

```python
import finetuner

finetuner.fit(
    embed_model,
    train_data=labeled_data
)
```

### 2️⃣ Have embedding model and unlabeled data

You have an `embed_model` to use, but no labeled data for finetuning this model. No worry, that's good enough already!
You can use Finetuner to interactive label data and train `embed_model` as below:

```python
import finetuner

finetuner.fit(
    embed_model,
    train_data=unlabeled_data,
    interactive=True
)
```

### 3️⃣ Have general model and labeled data

You have a `general_model` which does not output embeddings. Luckily you provide some `labeled_data` for training. No
worry, Finetuner can convert your model into an embedding model and train it via:

```python
import finetuner

finetuner.fit(
    general_model,
    train_data=labeled_data,
    to_embedding_model=True,
    output_dim=100
)
```

### 4️⃣ Have general model and unlabeled data

You have a `general_model` which is not for embeddings. Meanwhile, you don't have labeled data for training. But no
worries, Finetuner can help you train an embedding model with interactive labeling on-the-fly:

```python
import finetuner

finetuner.fit(
    general_model,
    train_data=unlabeled_data,
    interactive=True,
    to_embedding_model=True,
    output_dim=100
)
```

## Finetuning ResNet50 on CelebA

> ⚡ To get the best experience, you need a GPU machine to run this example. For CPU users, we provide [finetuning a MLP on FashionMNIST](https://finetuner.jina.ai/get-started/fashion-mnist/) and [finetuning a Bi-LSTM on CovidQA](https://finetuner.jina.ai/get-started/covid-qa/) that run out the box. Give them a try!

1. Download [CelebA dataset](https://static.jina.ai/celeba/celeba-img.zip) and decompress it to './img_align_celeba'
2. Finetuner accepts Jina `DocumentArray`/`DocumentArrayMemmap`, so we load CelebA data into this format using generator:
    ```python
    from jina.types.document.generators import from_files

    def data_gen():
        for d in from_files('./img_align_celeba/*.jpg', size=100, to_dataturi=True):
            d.convert_image_datauri_to_blob(color_axis=0)  # no need of tf
            yield d
    ```
3. Load pretrained ResNet50.
    - Pytorch
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
4. Starts the Finetuner
    ```python
    import finetuner
    
    finetuner.fit(
        model=model,
        interactive=True,
        train_data=data_gen,
        freeze=True,
        to_embedding_model=True,
        input_size=(3, 224, 224),
        output_dim=100
    )
    ```
5. Now the browser will open the Labeler UI. You can now label the data by mouse/keyboard. The model will get finetuned and improved as you are labeling. If you are running this example on a CPU machine, it can take up to 20 seconds for each labeling round.

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