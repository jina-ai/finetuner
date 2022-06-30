<p align="center">
<img src="https://github.com/jina-ai/finetuner/blob/main/docs/_static/finetuner-logo-ani.svg?raw=true" alt="Finetuner logo: Finetuner helps you to create experiments in order to improve embeddings on search tasks. It accompanies you to deliver the last mile of performance-tuning for neural search applications." width="150px">
</p>


<p align="center">
<b>Domain-specific fine-tuning for better embeddings on neural search tasks</b>
</p>

<p align=center>
<a href="https://pypi.org/project/finetuner/"><img src="https://img.shields.io/badge/Python-3.7%2B-blue alt="Python 3.7" title="Finetuner supports Python 3.7 and above"></a>
<a href="https://slack.jina.ai"><img src="https://img.shields.io/badge/Slack-2.2k%2B-blueviolet?logo=slack&amp;logoColor=white"></a>
</p>

<!-- start elevator-pitch -->

Fine-tuning with domain specific data can improve the performance on neural search tasks.
However, it is non-trivial as it requires a combination of expertise of deep learning and information retrieval.

Finetuner makes this procedure simpler, faster and performant by streamlining the workflow and handling all complexity and infrastructure on the cloud.
With Finetuner, you can easily make pre-trained models more performant and production ready.

📈**Performance boost**: Finetuner delivers SOTA performance on domain specific neural search applications.

🔱 **Simple yet powerful**: Easily access features such as 40+ mainstream loss functions, 10+ optimisers, layer pruning, weights freezing, dimensionality reduction, hard-negative mining, cross modality fine-tuning, distributed training. 

☁ **All-in-cloud**: Manage your runs, experiments and artifacts on Jina Cloud ([for free!](https://docs.google.com/forms/d/e/1FAIpQLSeoEhJM_TWMgZyEgJBBpf33JddcWQgXHNglNjVMIOvlLjk-4A/viewform)) without worrying about provisioning resources. You never have to worry about provisioning (cloud) resources! Finetuner handles all related complexity and infrastructure.

<!-- end elevator-pitch -->

## [Documentation](https://finetuner.jina.ai/)

## Benchmark

The following table demonstrates what you can expect from Finetuner:

<table>
<thead>
  <tr>
    <th>MODEL</th>
    <th>TASK</th>
    <th>METRIC</th>
    <th>PRETRAINED</th>
    <th>FINETUNED</th>
    <th>DELTA</th>
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
    <td>:arrow_up_small: 217%</td>
  </tr>

</tbody>
</table>

<sub><sup><a id="example-setup">[*]</a> All metrics evaluation on k@20, trained 5 epochs using Adam optimizer with learning rate of 1e-5.</sup></sub>

## Install

Requires Python 3.7+.

```bash
pip install -U finetuner
```

Noted: Starting from 0.5.0, Finetuner becomes cloud-based.
If you still want to use the last Finetuner release which runs locally, please install with:

```bash
pip install finetuner==0.4.1
```

We have backed up the 0.4.1 documentation in `docs/docs_41/` folder.
Check [this page](docs/docs_41/README.md) to render Finetuner 0.4.1 documentation locally.

## Get Started

The following code block describes how to fine-tune ResNet50 on [Totally Looks Like dataset](https://sites.google.com/view/totally-looks-like-dataset), which can be run as-is:
```python
import finetuner
from finetuner.callback import EvaluationCallback

finetuner.login()

run = finetuner.fit(
    model='resnet50',
    run_name='resnet50-tll-run',
    train_data='tll-train-da',
    callbacks=[EvaluationCallback(query_data='tll-eval-da')],
)
```

Fine-tuning might take some time until finish.
Once it is done, you can re-connect your run with:

```python
import finetuner

finetuner.login()

run = finetuner.get_run('resnet50-tll-run')
print(run.status())
print(run.logs())

run.save_artifact('resnet-tll')
```

It has the following steps:

  * Login to Finetuner: This is necessary if you'd like to run fine-tuning jobs with Finetuner in the cloud.
  * Start fine-tuning run: Select backbone model, training and evaluation data for your evaluation callback.
  * Monitor: Check the status and logs of the progress on your fine-tuning run.
  * Save model: If your fine-tuning run has successfully completed, save it for further use and integration.

### Next steps

- Take a look at the [walk through](https://finetuner.jina.ai/walkthrough/) documentation for an overview of how Finetuner works.
+ Get started with our example use-cases:
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
