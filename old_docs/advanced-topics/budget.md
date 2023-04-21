(budget)=
# {octicon}`database` How much data?

```{admonition} Read full blog
:class: hint
Please checkout [Fine-tuning with Low Budget and High Expectations](https://jina.ai/news/fine-tuning-with-low-budget-and-high-expectations/)
to read the full tech blog.
```

Fine-tuning takes a pre-trained model,
trained on a related task, and then further trains it for a new task.
Alternately, it can mean taking a model pre-trained for an open domain task, and further training it for a domain-specific one.
Compared to training from scratch, fine-tuning is a much more cost-efficient solution whenever it is feasible. But:

+ Exactly how much **data** do you need to get a good result?
+ Exactly how much **time** do you need to get good results?

## Experiments

We designed two experiments to quantitatively study how labeled data and training time affect fine-tuning performance.
For each experiment, we constructed three search tasks by fine-tuning three models.
We chose seven datasets, two of which are non-domain-specific public datasets, to ensure the generality of our experiment.

We measured the performance of the fine-tuned models by evaluating their ability to perform search tasks, as measured by Mean Reciprocal Rank (mRR), Recall, and Mean Average Precision (mAP).
These metrics are calculated using the top 20 results of each search in the validation subset held out from each dataset.

### How much labeled data is needed?

We gradually increase the amount of labeled data fed to Finetuner from 100 items to 100,000 and see how this affects performance on the metrics described in the previous section.

In the figures below, the X-axis represents the amount of labeled data, and the Y-axis represents the relative improvement over the pre-trained model. The higher, the better.

...             |  ...
:-------------------------:|:-------------------------:
![text-text-quora](https://jina-ai-gmbh.ghost.io/content/images/2022/12/Text-to-text-search-on-QuoraQA--3-.svg)  |  ![text-text-clinc](https://jina-ai-gmbh.ghost.io/content/images/2022/12/Text-to-text-search-on-Clinc150--3-.svg)
![image-image-tll](https://jina-ai-gmbh.ghost.io/content/images/2022/12/Image-to-image-search-on-Totally-looks-like.svg) | ![image-image-celeba](https://jina-ai-gmbh.ghost.io/content/images/2022/12/Image-to-image-search-on-Celeba--4-.svg)
![image-image-flickr30k](https://jina-ai-gmbh.ghost.io/content/images/2022/12/Text-to-image-search-on-Flickr30K--5-.svg) | ![image-image-coco](https://jina-ai-gmbh.ghost.io/content/images/2022/12/Text-to-image-search-on-CoCoCaptions--4-.svg)

These results are promising but not particularly surprising.
Performance improves with more labeled data on nearly all tasks and all datasets, more for some tasks and datasets than for others.
However, the only conclusion we can draw from these figures is that the Finetuner works as advertised. So far so good.

We further calculate the return on investment (ROI),
by dividing the relative improvement (a proxy for net profit) by the amount of labeled data (a proxy for investment cost).
**This is useful because it indicates the point at which adding more data is producing diminishing returns.**

In the figures below, the X-axis represents the amount of labeled data, and the Y-axis represents the ROI per labeled data item. The higher, the better.
In particular, `ROI=0` means adding new labeled data at that point no longer contributes to any improvement.

...             |  ...
:-------------------------:|:-------------------------:
![text-text-quora](https://jina-ai-gmbh.ghost.io/content/images/2022/12/Text-to-text-search-on-QuoraQA--7-.svg)  |  ![text-text-clinc](https://jina-ai-gmbh.ghost.io/content/images/2022/12/Text-to-text-search-on-Clinc150--7-.svg)
![image-image-tll](https://jina-ai-gmbh.ghost.io/content/images/2022/12/Image-to-image-search-on-Totally-looks-like--1-.svg) | ![image-image-celeba](https://jina-ai-gmbh.ghost.io/content/images/2022/12/Image-to-image-search-on-Celeba--5-.svg)
![image-image-flickr30k](https://jina-ai-gmbh.ghost.io/content/images/2022/12/Text-to-image-search-on-Flickr30K--6-.svg) | ![image-image-coco](https://jina-ai-gmbh.ghost.io/content/images/2022/12/Text-to-image-search-on-CoCoCaptions--5-.svg)

Surprisingly, we can see that the ROI per unit of new labeled data starts to drop almost immediately. We expected that it would eventually decrease, but this is an unexpected result.

### How much time is needed?

To measure the value of added training time, we fixed the amount of new labeled data to 1000 items, and then we gradually increased the number of training epochs from 1 to 10.
At each increase, we measure improvement over the pre-trained model and calculate the ROI.
For these experiments, the ROI is calculated by dividing the relative improvement by the elapsed time in seconds.
This means that when `ROI=0`, adding training time no longer improves performance.

...            |  ...
:-------------------------:|:-------------------------:
![text-text-quora](https://jina-ai-gmbh.ghost.io/content/images/2022/12/Text-to-text-search-on-QuoraQA--4-.svg)  |  ![text-text-clinc](https://jina-ai-gmbh.ghost.io/content/images/2022/12/Text-to-text-search-on-Clinc150--4-.svg)
![image-image-tll](https://jina-ai-gmbh.ghost.io/content/images/2022/12/Image-to-image-search-on-Totally-look-like--2-.svg) | ![image-image-celeba](https://jina-ai-gmbh.ghost.io/content/images/2022/12/Image-to-image-search-on-Celeba--2-.svg)
![image-image-flickr30k](https://jina-ai-gmbh.ghost.io/content/images/2022/12/Text-to-image-search-on-Flickr30K--3-.svg) | ![image-image-coco](https://jina-ai-gmbh.ghost.io/content/images/2022/12/Text-to-image-search-on-CocoCaptions--2-.svg)

We knew in advance that adding more time does not guarantee any improvement at all.
It can, in fact, reduce performance due to the overfitting problem.
Some models (e.g. CLIP) are more prone to overfitting than others.
In principle, if we keep training with the same 1000 data points over and over, we are guaranteed to overfit on the data and the overall performance will drop.

Let's look at the ROI curves.

...             |  ...
:-------------------------:|:-------------------------:
![text-text-quora](https://jina-ai-gmbh.ghost.io/content/images/2022/12/Text-to-text-search-on-QuoraQA--5-.svg)  |  ![text-text-clinc](https://jina-ai-gmbh.ghost.io/content/images/2022/12/Text-to-text-search-on-Clinc150--9-.svg)
![image-image-tll](https://jina-ai-gmbh.ghost.io/content/images/2022/12/Image-to-image-search-on-Totally-look-like--3-.svg) | ![image-image-celeba](https://jina-ai-gmbh.ghost.io/content/images/2022/12/Image-to-image-search-on-Celeba--3-.svg)
![image-image-flickr30k](https://jina-ai-gmbh.ghost.io/content/images/2022/12/Text-to-image-search-on-Flickr30K--4-.svg) | ![image-image-coco](https://jina-ai-gmbh.ghost.io/content/images/2022/12/Text-to-image-search-on-CocoCaptions--3-.svg)

The ROI drops immediately after the first epoch of fine-tuning.
Unlike in the last experiment, where ROI approached zero but stayed positive when increasing the number of epochs, here, the ROI on added time can go negative due to the overfitting problem!

## Summary

What does this mean for users looking to maximize gains and minimize costs?

+ Many state-of-the-art deep neural networks are capable of few-shot learning. They are quick learners and can make large improvements with only a few hundred items of labeled data and only a few minutes of training time. You might have thought that deep neural network training requires millions of data items and a week of runtime, but we have shown in these examples how that stereotype does not hold up to reality.
+ Because they can learn so much, so fast, from so little data, ROI drops quickly as you put more time and data into fine-tuning. In the experiments above, ROI shrinks by 70% from its highest value after 500 labeled data items or 600 added seconds of GPU training time. Further investment beyond a few hundred items of training data and very minimal training time may not pay off as well as you would like.