<br><br>

<p align="center">
<img src="https://github.com/jina-ai/finetuner/blob/main/docs/_static/finetuner-logo-ani.svg?raw=true" alt="Finetuner logo: Finetuner helps you to create experiments in order to improve embeddings on search tasks. It accompanies you to deliver the last mile of performance-tuning for neural search applications." width="150px">
</p>


<p align="center">
<b>Task-oriented finetuning for better embeddings on neural search</b>
</p>

<p align=center>
<a href="https://pypi.org/project/finetuner/"><img alt="PyPI" src="https://img.shields.io/pypi/v/finetuner?label=Release&style=flat-square"></a>
<a href="https://codecov.io/gh/jina-ai/finetuner"><img alt="Codecov branch" src="https://img.shields.io/codecov/c/github/jina-ai/finetuner/main?logo=Codecov&logoColor=white&style=flat-square"></a>
<a href="https://pypistats.org/packages/finetuner"><img alt="PyPI - Downloads from official pypistats" src="https://img.shields.io/pypi/dm/finetuner?style=flat-square"></a>
<a href="https://slack.jina.ai"><img src="https://img.shields.io/badge/Slack-3.6k-blueviolet?logo=slack&amp;logoColor=white&style=flat-square"></a>
</p>

<!-- start elevator-pitch -->

Fine-tuning is an effective way to improve performance on neural search tasks. However, it can be very difficult for
many deep learning engineers to do.

Jina AI's Finetuner makes fine-tuning easier and faster by streamlining the workflow and handling all complexity and 
infrastructure in the cloud. With Finetuner, one can easily enhance the performance of pre-trained models, making them 
production-ready without buying expensive hardware.

📈 **Performance promise**: enhance the performance of pre-trained models and deliver state-of-the-art performance on 
domain-specific neural search applications.

🔱 **Simple yet powerful**: easy access to 40+ mainstream loss functions, 10+ optimisers, layer pruning, weight 
freezing, dimensionality reduction, hard-negative mining, cross-modal models, and distributed training. 

☁ **All-in-cloud**: train using our free GPU infrastructure, manage runs, experiments and artifacts on Jina AI Cloud without worrying 
about resource availability, complex integration, or scalable infrastructure.

<!-- end elevator-pitch -->

## [Documentation](https://finetuner.jina.ai/)

## Benchmarks

<table>
<thead>
  <tr>
    <th>Model</th>
    <th>Task</th>
    <th>Metric</th>
    <th>Pretrained</th>
    <th>Finetuned</th>
    <th>Delta</th>
    <th>Run it!</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td rowspan="2">BERT</td>
    <td rowspan="2"><a href="https://www.kaggle.com/c/quora-question-pairs">Quora</a> Question Answering</td>
    <td>mRR</td>
    <td>0.835</td>
    <td>0.967</td>
    <td><span style="color:green">15.8%</span></td>
    <td rowspan="2"><p align=center><a href="https://colab.research.google.com/drive/1Ui3Gw3ZL785I7AuzlHv3I0-jTvFFxJ4_?usp=sharing"><img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg"></a></p></td>
  </tr>
  <tr>
    <td>Recall</td>
    <td>0.915</td>
    <td>0.963</td>
    <td><span style="color:green">5.3%</span></td>
  </tr>
  <tr>
    <td rowspan="2">ResNet</td>
    <td rowspan="2">Visual similarity search on <a href="https://sites.google.com/view/totally-looks-like-dataset">TLL</a></td>
    <td>mAP</td>
    <td>0.110</td>
    <td>0.196</td>
    <td><span style="color:green">78.2%</span></td>
    <td rowspan="2"><p align=center><a href="https://colab.research.google.com/drive/1QuUTy3iVR-kTPljkwplKYaJ-NTCgPEc_?usp=sharing"><img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg"></a></p></td>
  </tr>
  <tr>
    <td>Recall</td>
    <td>0.249</td>
    <td>0.460</td>
    <td><span style="color:green">84.7%</span></td>
  </tr>
  <tr>
    <td rowspan="2">CLIP</td>
    <td rowspan="2"><a href="https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html">Deep Fashion</a> text-to-image search</td>
    <td>mRR</td>
    <td>0.575</td>
    <td>0.676</td>
    <td><span style="color:green">17.4%</span></td>
    <td rowspan="2"><p align=center><a href="https://colab.research.google.com/drive/1yKnmy2Qotrh3OhgwWRsMWPFwOSAecBxg?usp=sharing"><img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg"></a></p></td>
  </tr>
  <tr>
    <td>Recall</td>
    <td>0.473</td>
    <td>0.564</td>
    <td><span style="color:green">19.2%</span></td>
  </tr>

</tbody>
</table>

<sub><sup>All metrics were evaluated for k@20 after training for 5 epochs using the Adam optimizer with learning rates of 1e-4 for ResNet, 1e-7 for CLIP and 1e-5 for the BERT models.</sup></sub>

<!-- start install-instruction -->

## Installation

Make sure you have Python 3.7+ installed. Finetuner can be installed via `pip` by executing:

```bash
pip install -U finetuner
```

If you want to encode `docarray.DocumentArray` objects with the `finetuner.encode` function, you will need to install 
`"finetuner[full]"`. This includes a number of additional dependencies, which are necessary for encoding: torch, 
torchvision and OpenCLIP:

```bash
pip install "finetuner[full]"
```

<!-- end install-instruction -->

> ⚠️ Starting with version 0.5.0, Finetuner computing is performed on Jina AI Cloud. The last local version is `0.4.1`. 
> This version is still available for installation via `pip`. See [Finetuner git tags and releases](https://github.com/jina-ai/finetuner/releases).




  
## Get Started

The following code snippet describes how to fine-tune ResNet50 on the [_Totally Looks Like_ dataset](https://sites.google.com/view/totally-looks-like-dataset). 
You can run it as-is. The model and training data are already hosted in Jina AI Cloud and the Finetuner server will 
download them automatically.
(NB: If there is already a run called `resnet50-tll-run`, choose a different run-name in the code below.)

```python
import finetuner
from finetuner.callback import EvaluationCallback

finetuner.login()

run = finetuner.fit(
    model='resnet50',
    run_name='resnet50-tll-run',
    train_data='tll-train-data',
    callbacks=[
        EvaluationCallback(
            query_data='tll-test-query-data',
            index_data='tll-test-index-data',
        )
    ],
)
```
This code snippet describes the following steps:

* Login to Jina AI Cloud.
* Select backbone model, training and evaluation data for your evaluation callback.
* Start the cloud run.

You can also pass data to Finetuner as a CSV file or a `DocumentArray` object, as described [in the Finetuner documentation](https://finetuner.jina.ai/walkthrough/create-training-data/).  

Depending on the data, task, model, hyperparameters, fine-tuning might take some time to finish. You can leave your jobs 
to run on the Jina AI Cloud, and later reconnect to them, using code like this below:

```python
import finetuner

finetuner.login()

run = finetuner.get_run('resnet50-tll-run')

for log_entry in run.stream_logs():
    print(log_entry)

run.save_artifact('resnet-tll')
```

This code logs into Jina AI Cloud, then connects to your run by name. After that, it does the following:
  * Monitors the status of the run and prints out the logs.
  * Saves the model once fine-tuning is done.

## Using Finetuner to encode

Finetuner has interfaces for using models to do encoding:

```python
import finetuner
from docarray import Document, DocumentArray

da = DocumentArray([Document(uri='~/Pictures/your_img.png')])

model = finetuner.get_model('resnet-tll')
finetuner.encode(model=model, data=da)

da.summary()
```

When encoding, you can provide data either as a DocumentArray or a list. Since the modality of your input data can be inferred from the model being used, there is no need to provide any additional information besides the content you want to encode. When providing data as a list, the `finetuner.encode` method will return a `np.ndarray` of embeddings, instead of a `docarray.DocumentArray`:

```python
import finetuner
from docarray import Document, DocumentArray

images = ['~/Pictures/your_img.png']

model = finetuner.get_model('resnet-tll')
embeddings = finetuner.encode(model=model, data=images)
```

## Training on your own data

If you want to train a model using your own dataset instead of one on the Jina AI Cloud, you can provide labeled data in a CSV file.

A CSV file is a tab or comma-delimited plain text file. For example:

```plaintext
This is an apple    apple_label
This is a pear      pear_label
...
```
The file should have two columns: The first for the data and the second for the category label.

You can then provide a path to a CSV file as training data for Finetuner:

```python
run = finetuner.fit(
    model='bert-base-cased',
    run_name='bert-my-own-run',
    train_data='path/to/some/data.csv',
)
```
More information on providing your own training data is found in the [Prepare Training Data](https://finetuner.jina.ai/walkthrough/create-training-data/) section of the [Finetuner documentation](https://finetuner.jina.ai/).



### Next steps

- Take the [walkthrough](https://finetuner.jina.ai/walkthrough/) and submit your first fine-tuning job.
- Try out different search tasks:
  - [Text-to-Text Search via BERT](https://finetuner.jina.ai/notebooks/text_to_text/)
  - [Image-to-Image Search via ResNet50](https://finetuner.jina.ai/notebooks/image_to_image/)
  - [Text-to-Image Search via CLIP](https://finetuner.jina.ai/notebooks/text_to_image/)

[Read our documentation](https://finetuner.jina.ai/) to learn more about what Finetuner can do.

<!-- start support-pitch -->
## Support

- Use [Discussions](https://github.com/jina-ai/finetuner/discussions) to talk about your use cases, questions, and
  support queries.
- Join our [Slack community](https://slack.jina.ai) and chat with other Jina AI community members about ideas.
- Join our [Engineering All Hands](https://youtube.com/playlist?list=PL3UBBWOUVhFYRUa_gpYYKBqEAkO4sxmne) meet-up to discuss your use case and learn Jina AI new features.
    - **When?** The second Tuesday of every month
    - **Where?**
      Zoom ([see our public events calendar](https://calendar.google.com/calendar/embed?src=c_1t5ogfp2d45v8fit981j08mcm4%40group.calendar.google.com&ctz=Europe%2FBerlin)/[.ical](https://calendar.google.com/calendar/ical/c_1t5ogfp2d45v8fit981j08mcm4%40group.calendar.google.com/public/basic.ics))
      and [live stream on YouTube](https://youtube.com/c/jina-ai)
- Subscribe to the latest video tutorials on our [YouTube channel](https://youtube.com/c/jina-ai)

## Join Us

Finetuner is backed by [Jina AI](https://jina.ai) and licensed under [Apache-2.0](./LICENSE). 

[We are actively hiring](https://jobs.jina.ai) AI engineers and solution engineers to build the next generation of
open-source AI ecosystems.

<!-- end support-pitch -->
