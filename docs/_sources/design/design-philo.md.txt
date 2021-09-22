(design-philo)=
# Design Philosophy

```{caution}
This section is not meant to be public. It is a collection of my thoughts on the high-level design & positioning of the Finetuner project in the Jina AI landscape. Please do not share any content below.

Some of the paragraph may be published later after my revision.
```

## Why Finetuner?

### Backstory

When I was working on the Yahaha case (3D mesh search), I leveraged Jina and quickly built the first solution in two days. After establishing of the pipeline and API, improving the search accuracy becomes natuarally the priority. However, I often times found myself stucking with the top-K result page and had nowhere to continue: should I retrain the model and expect everything to work better? Or is the current embedding good enough and I just need to "finetune" it a bit? I often wish to have an interface, where I can just click mouse and rearrange the search results around, then click the "tune" button, and the model just gets improved.

This motivates the Finetuner project.

### Transfer learning

Generally, training from scratch is [not cost-effective for many reasons I pointed out in this post](https://hanxiao.io/2019/07/29/Generic-Neural-Elastic-Search-From-bert-as-service-and-Go-Way-Beyond/?highlight=body%20%3E%20div.wrap%20%3E%20main%20%3E%20div%20%3E%20article%20%3E%20div.post-content%20%3E%20p:nth-child(20)). There is no need to learn common knowledge such as edges, textures, color, grammars, syntax again and again. If you work at Jina AI, then you should be very familiar with **Transfer Learning**, with its idea of decoupling the classic machine learning training into the **pretraining** and the **finetuning** parts. You should be also very well-aware of how it influrences the company's product landscape from the day one.

### Last mile delivery 

I see Finetuner project as the last mile delivery of Jina.

If you think about Jina users, the most value-adding part for them is Jina allows _anyone_ to build cloud-native neural search app in _minutes_. Yes, literally minutes. The time-saving and easy-to-use is attractive for many Jina developers. With continuous development on Jina core and the readiness of Jina Hub, the easy-to-use part is obvious.

In other words, Jina + Jina Hub solve the zero-to-one problem for building neural search applications.

So what about one-to-N? What is the next step after having an okayish-performant Jina app? How to reach from 70% accuracy to 99% accuracy? What can _we_ provide for this last mile delivery?

That's the purpose of Finetuner, 
- to get the last mile done;
- to keep users in our ecosystem;
- to be a potential monetization feature for mission-critical apps.

## Positioning of Finetuner

In our company landscape, Finetuner is a part of our "Ecosystem" block.

```{figure} landscape.png
:align: center
```

Though being positioned inside the "Ecosystem", Finetuner is not an independent product: it shares the same interface as Jina core; it leverages the Executors inside Jina Hub; it creates synergies with existing products and multiplies their value. This will affect the design decision of Finetuner, as I shall explain in the {ref}`next chapter <design-decision>`.

Let's zoom out a bit and look at [the global picture of AI supply chain](https://hanxiao.io/2019/07/29/Generic-Neural-Elastic-Search-From-bert-as-service-and-Go-Way-Beyond/?highlight=body%20%3E%20div.wrap%20%3E%20main%20%3E%20div%20%3E%20article%20%3E%20div.post-content%20%3E%20img:nth-child(26)) that I mentioned earlier. My points in that article are still valid today:
- Fewer and fewer people will work on building models from scratch, they are responsible for delivering pretrained models.
- Most developers will work on making sense of those pretrained models, applying them in every niche domain.

I don't want to talk about AI research vs. AI engineering here. I want to highlight one thing in this supply chain: **the tooling.**

```{figure} supply-chain.png
:align: center
```

Let me pose some questions, take a minute and think about them. Then, unfold my answer to them. 

```{dropdown} What is the tools for developing a DNN model from scratch?

Tensorflow or Pytorch. 

They have became the defacto tooling for creating a new model. Researchers use them to create pretrained models.
```

```{dropdown} What is the tools for getting pretrained model?

Huggingface.

I belive Huggingface solves one problem correctly: **the strong and universal need of fetching pretrained models.** This makes HF popular among developers. Think about the days before HF, how would you get a pretrained model? Browse Google/Microsoft/Cambridge website; download a zip file; decompress it and set up enviroment; follow the arxiv paper and load model by hand; and finally if you change to a new enviroment, you have to repeat the whole procedure again!

With HF, the only thing you need to do is `from huggingface import bert`
```

```{dropdown} What is the tools for tuning pretrained models on particular domain?

Nothing, yet! 

And that's the chance of Finetuner project. Finetuner will fill the blank of model tuning in the search domain.
 
Why not Tensorflow/Pytorch again here? They are too heavy and too complicated for finetuning task. Plus, I belive a good finetuner should have:
- extremely simple interface (as the users here are *not* exposed to the full knowledge of deep learning anymore!)
- exteremly easy way to incorpate human feedback and domain knowledge.
```

Finetuner is our response to the new AI supply chain. I believe finetuning is such a strong and common need in the age of transfer learning. If you don't see it now, you will see it in two years (probably less).


