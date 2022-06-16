<p align="center">
<img src="https://github.com/jina-ai/finetuner/blob/main/docs/_static/finetuner-logo-ani.svg?raw=true" alt="Finetuner logo: Finetuner helps you to create experiments in order to improve embeddings on search tasks. It accompanies you to deliver the last mile of performance-tuning for neural search applications." width="150px">
</p>


<p align="center">
<b>Fine-tuning embeddings on domain specific data for better performance on neural search tasks.</b>
</p>

<p align=center>
<a href="https://pypi.org/project/finetuner/"><img src="https://img.shields.io/badge/Python-3.9%2B-blue alt="Python 3.9" title="Finetuner supports Python 3.9 and above"></a>
<a href="https://slack.jina.ai"><img src="https://img.shields.io/badge/Slack-2.2k%2B-blueviolet?logo=slack&amp;logoColor=white"></a>
</p>

<!-- start elevator-pitch -->

Fine-tuning deep neural networks (DNNs) significantly improves performance on domain specific neural search tasks.
However, fine-tuning for neural search is not trivial, as it requires a combination of expertise in ML and Information Retrieval.
Finetuner makes fine-tuning simple and fast by handling all related complexity and infrastructure in the cloud. With Finetuner, you can easily make models more performant and production ready.

📈**Performance boost**: Finetuner significantly increases the performance of pretrained models on domain specific neural search applications.

🔱 **Simple yet powerful**: Interacting with Finetuner is simple and seamless, and also supports rich features such as selections of different loss functions, e.g. 
siamese/triplet loss, metric learning, layer pruning, weights freezing, dimensionality reduction, and much more.

☁ **Fine-tune in the cloud**: Finetuner runs your fine-tuning jobs in the cloud. You never have to worry about provisioning (cloud) resources! Finetuner handles all related complexity and infrastructure.

<!-- end elevator-pitch -->

## What is the purpose of Finetuner?

Finetuner enables performance gains on domain specific neural search tasks by fine-tuning models in the cloud. We have conducted experiments on various neural search tasks in different domains to illustrate these performance improvements.

Finetuner also aims to make fine-tuning simple and fast. When interacting with Finetuner, the API takes care of all your fine-tuning jobs in the cloud. This only requires a few lines of code from you, as demonstrated in [below](#fine-tuning-resnet50-on-totally-looks-like-dataset).

## How does it work?

<img src="https://github.com/jina-ai/finetuner/blob/docs-update-readme/docs/_static/finetuner-client-journey.svg?raw=true" title="Finetuner Client user journey.">


## Install

Requires Python 3.7+ installed on Linux/MacOS.

```bash
pip install -U finetuner-client
```


## Fine-tuning ResNet50 on Totally Looks Like dataset

```python
import finetuner
from finetuner.callback import EvaluationCallback

finetuner.login()

finetuner.create_experiment(name='tll-experiment')

run = finetuner.fit(
        model='resnet50',
        train_data='resnet-tll-train-data',
        callbacks=[EvaluationCallback(query_data='resnet-tll-eval-data')],
)

print(run.status())
print(run.logs())

run.save_model('resnet-tll')
```

This minimal example code starts a fine-tuning run with only the necessary arguments. It has the following steps:

  * Login to Finetuner: This is necessary if you'd like to run fine-tuning jobs with Finetuner in the cloud.
  * Create experiment: This experiment will contain various runs with different configurations.
  * Start fine-tuning run: Select backbone model, training and evaluation data for your evaluation callback.
  * Monitor: Check the status and logs of the progress on your fine-tuning run.
  * Save model: If your fine-tuning run has successfully completed, save it for further use and integration.


<!-- start support-pitch -->
## Support

- Take a look at the [step by step](https://ft-docs-polish--jina-docs.netlify.app/2_step_by_step/) documentation for an overview of how Finetuner works.
- Get started with our example use-cases in the [Finetuner in action](https://ft-docs-polish--jina-docs.netlify.app/3_finetuner_in_action/) section.
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