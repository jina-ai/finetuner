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

Fine-tuning is an effective way to improve performance on [neural search](https://jina.ai/news/what-is-neural-search-and-learn-to-build-a-neural-search-engine/) tasks.
However, setting up and performing fine-tuning can be very time-consuming and resource-intensive.

Jina AI's Finetuner makes fine-tuning easier and faster by streamlining the workflow and handling all the complexity and infrastructure in the cloud.
With Finetuner, you can easily enhance the performance of pre-trained models,
making them production-ready [without extensive labeling](https://jina.ai/news/fine-tuning-with-low-budget-and-high-expectations/) or expensive hardware.

🎏 **Better embeddings**: Create high-quality embeddings for semantic search, visual similarity search, cross-modal text<->image search, recommendation systems,
clustering, duplication detection, anomaly detection, or other uses.

⏰ **Low budget, high expectations**: Bring considerable improvements to model performance, making the most out of as little as a few hundred training samples, and finish fine-tuning in as little as an hour.

📈 **Performance promise**: Enhance the performance of pre-trained models so that they deliver state-of-the-art performance on 
domain-specific applications.

🔱 **Simple yet powerful**: Easy access to 40+ mainstream loss functions, 10+ optimizers, layer pruning, weight
freezing, dimensionality reduction, hard-negative mining, cross-modal models, and distributed training. 

☁ **All-in-cloud**: Train using our free GPU infrastructure, manage runs, experiments, and artifacts on Jina AI Cloud
without worrying about resource availability, complex integration, or infrastructure costs.

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
  <tr>
    <td rowspan="2">M-CLIP</td>
    <td rowspan="2"><a href="https://xmrec.github.io/">Cross market</a> product recommendation (German)</td>
    <td>mRR</td>
    <td>0.430</td>
    <td>0.648</td>
    <td><span style="color:green">50.7%</span></td>
    <td rowspan="2"><p align=center><a href="https://colab.research.google.com/drive/10Wldbu0Zugj7NmQyZwZzuorZ6SSAhtIo"><img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg"></a></p></td>
  </tr>
  <tr>
    <td>Recall</td>
    <td>0.247</td>
    <td>0.340</td>
    <td><span style="color:green">37.7%</span></td>
  </tr>
  <tr>
    <td rowspan="2">PointNet++</td>
    <td rowspan="2"><a href="https://modelnet.cs.princeton.edu/">ModelNet40</a> 3D Mesh Search</td>
    <td>mRR</td>
    <td>0.791</td>
    <td>0.891</td>
    <td><span style="color:green">12.7%</span></td>
    <td rowspan="2"><p align=center><a href="https://colab.research.google.com/drive/1lIMDFkUVsWMshU-akJ_hwzBfJ37zLFzU?usp=sharing"><img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg"></a></p></td>
  </tr>
  <tr>
    <td>Recall</td>
    <td>0.154</td>
    <td>0.242</td>
    <td><span style="color:green">57.1%</span></td>
  </tr>

</tbody>
</table>

<sub><sup>All metrics were evaluated for k@20 after training for 5 epochs using the Adam optimizer with learning rates of 1e-4 for ResNet, 1e-7 for CLIP and 1e-5 for the BERT models, 5e-4 for PointNet++</sup></sub>

<!-- start install-instruction -->

## Install

Make sure you have Python 3.8+ installed. Finetuner can be installed via `pip` by executing:

```bash
pip install -U finetuner
```

If you want to encode `docarray.DocumentArray` objects with the `finetuner.encode` function, you need to install 
`"finetuner[full]"`. This includes a number of additional dependencies, which are necessary for encoding: Torch, 
Torchvision and OpenCLIP:

```bash
pip install "finetuner[full]"
```

<!-- end install-instruction -->

> ⚠️ Starting with version 0.5.0, Finetuner computing is performed on Jina AI Cloud. The last local version is `0.4.1`. 
> This version is still available for installation via `pip`. See [Finetuner git tags and releases](https://github.com/jina-ai/finetuner/releases).

<!-- start finetuner-articles -->
## Articles about Finetuner

Check out our published blogposts and tutorials to see Finetuner in action!

- [Fine-tuning with Low Budget and High Expectations](https://jina.ai/news/fine-tuning-with-low-budget-and-high-expectations/)
- [Hype and Hybrids: Search is more than Keywords and Vectors](https://jina.ai/news/hype-and-hybrids-multimodal-search-means-more-than-keywords-and-vectors-2/)
- [Improving Search Quality for Non-English Queries with Fine-tuned Multilingual CLIP Models](https://jina.ai/news/improving-search-quality-non-english-queries-fine-tuned-multilingual-clip-models/)
- [How Much Do We Get by Finetuning CLIP?](https://jina.ai/news/applying-jina-ai-finetuner-to-clip-less-data-smaller-models-higher-performance/)

<!-- end finetuner-articles -->

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
