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

Fine-tuning is an effective way to improve the performance on neural search tasks. However, it is non-trivial for many deep learning engineers.

Finetuner makes fine-tuning easier, faster and performant by streamlining the workflow and handling all complexity and infrastructure on the cloud.
With Finetuner, one can easily uplift pre-trained models to be more performant and production ready.

📈 **Performance promise**: uplift pretrained model and deliver SOTA performance on domain-specific neural search applications.

🔱 **Simple yet powerful**: easy access to 40+ mainstream losses, 10+ optimisers, layer pruning, weights freezing, dimensionality reduction, hard-negative mining, cross-modal model, distributed training. 

☁ **All-in-cloud**: instant training with our free GPU ([Apply here for free!](https://docs.google.com/forms/d/e/1FAIpQLSeoEhJM_TWMgZyEgJBBpf33JddcWQgXHNglNjVMIOvlLjk-4A/viewform)); manage runs, experiments and artifacts on Jina Cloud without worrying about provisioning resources, integration complexity and infrastructure.

<!-- end elevator-pitch -->

## [Documentation](https://finetuner.jina.ai/)

## Benchmark

<table>
<thead>
  <tr>
    <th>Model</th>
    <th>Task</th>
    <th>Metric</th>
    <th>Pretrained</th>
    <th>Finetuned</th>
    <th>Delta</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td rowspan="2">BERT</td>
    <td rowspan="2"><a href="https://www.kaggle.com/c/quora-question-pairs">Quora</a> Question Answering</td>
    <td>mRR</td>
    <td>0.835</td>
    <td>0.967</td>
    <td>:arrow_up_small: 15.8%</td>
  </tr>
  <tr>
    <td>Recall</td>
    <td>0.915</td>
    <td>0.963</td>
    <td>:arrow_up_small: 5.3%</td>
  </tr>
  <tr>
    <td rowspan="2">ResNet</td>
    <td rowspan="2">Visual similarity search on <a href="https://sites.google.com/view/totally-looks-like-dataset">TLL</a></td>
    <td>mAP</td>
    <td>0.102</td>
    <td>0.166</td>
    <td>:arrow_up_small: 62.7%</td>
  </tr>
  <tr>
    <td>Recall</td>
    <td>0.235</td>
    <td>0.372</td>
    <td>:arrow_up_small: 58.3%</td>
  </tr>
  <tr>
    <td rowspan="2">CLIP</td>
    <td rowspan="2"><a href="https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html">Deep Fashion</a> text-to-image search</td>
    <td>mRR</td>
    <td>0.289</td>
    <td>0.488</td>
    <td>:arrow_up_small: 69.9%</td>
  </tr>
  <tr>
    <td>Recall</td>
    <td>0.109</td>
    <td>0.346</td>
    <td>:arrow_up_small: 217.0%</td>
  </tr>

</tbody>
</table>

<sub><sup><a id="example-setup">[*]</a> All metrics evaluation on k@20, trained 5 epochs using Adam optimizer with learning rate of 1e-5.</sup></sub>

<!-- start install-instruction -->

## Install

Make sure you have Python 3.7+ installed.
Finetuner can be installed via pip by executing:

```bash
pip install -U finetuner
```

If you want to encode `docarray.DocumentArray` objects with the `finetuner.encode` function, you need to install `"finetuner[full]"`.
In this case, some extra dependencies are installed which are necessary to do the inference, e.g., torch, torchvision, and open clip:

```bash
pip install "finetuner[full]"
```

<!-- end install-instruction -->

> From 0.5.0, Finetuner computing is hosted on Jina Cloud. THe last local version is `0.4.1`, one can install it via pip or check out [git tags/releases here](https://github.com/jina-ai/finetuner/releases).




  
## Get Started

The following code snippet describes how to fine-tune ResNet50 on [Totally Looks Like dataset](https://sites.google.com/view/totally-looks-like-dataset), it can be run as-is:

```python
import finetuner
from finetuner.callback import EvaluationCallback

finetuner.login()

run = finetuner.fit(
    model='resnet50',
    run_name='resnet50-tll-run',
    train_data='tll-train-da',
    callbacks=[
        EvaluationCallback(
            query_data='tll-test-query-da',
            index_data='tll-test-index-da',
        )
    ],
)
```

Fine-tuning might take 5 minute to finish. You can later re-connect your run with:

```python
import finetuner

finetuner.login()

run = finetuner.get_run('resnet50-tll-run')

for msg in run.stream_logs():
    print(msg)

run.save_artifact('resnet-tll')
```

Specifically, the code snippet describes the following steps:

  * Login to Finetuner ([Get free access here!](https://docs.google.com/forms/d/e/1FAIpQLSeoEhJM_TWMgZyEgJBBpf33JddcWQgXHNglNjVMIOvlLjk-4A/viewform))
  * Select backbone model, training and evaluation data for your evaluation callback.
  * Start the cloud run.
  * Monitor the status: check the status and logs of the run.
  * Save model for further use and integration.


Finally, you can use the model to encode images:

```python
import finetuner
from docarray import Document, DocumentArray

da = DocumentArray([Document(uri='~/Pictures/your_img.png')])

model = finetuner.get_model('resnet-tll')
finetuner.encode(model=model, data=da)

da.summary()
```

### Next steps

- Take the [walkthrough](https://finetuner.jina.ai/walkthrough/) and submit your first fine-tuning job.
- Try on different search tasks:
  - [Text-to-Text Search via BERT](https://finetuner.jina.ai/tasks/text-to-text/)
  - [Image-to-Image Search via ResNet50](https://finetuner.jina.ai/tasks/image-to-image/)
  - [Text-to-Image Search via CLIP](https://finetuner.jina.ai/tasks/text-to-image/)

Intrigued? That's only scratching the surface of what Finetuner is capable of. [Read our docs to learn more](https://finetuner.jina.ai/).

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

Finetuner is backed by [Jina AI](https://jina.ai) and licensed under [Apache-2.0](./LICENSE). [We are actively hiring](https://jobs.jina.ai) AI engineers, solution engineers to build the next neural search ecosystem in opensource.

<!-- end support-pitch -->
