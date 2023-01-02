(budget)=
# {octicon}`database` How much data?

## Motivation

Fine-tuning is a transfer learning technique developed as part of the Deep Learning revolution in artificial intelligence.
Instead of learning a new task from scratch,
fine-tuning takes a pre-trained model,
trained on a related task, and then further trains it for the new task.
Alternately, it can mean taking a model pre-trained for an open domain task, and further training it for a domain-specific one.
Compared to training from scratch, fine-tuning is a much more cost-efficient solution whenever it is feasible. It requires:

+ **less labeled data**: as there is no need to learn everything all over again. All the training is devoted to acquiring domain-specific knowledge.
+ **less time to train**: since the number of variables is much smaller and most layers in the deep neural network freeze during fine-tuning.

But:

+ **Exactly how much data do you need to get a good result?** One labeled data point? Ten? One thousand? Ten thousand?
+ **Exactly how much time do you need to get good results?** One minute of fine-tuning? An hour? A day? A week?

## Experiments

We designed two experiments to quantitatively study how labeled data and training time affect fine-tuning performance.
For each experiment, we construct three search tasks by fine-tuning three deep neural networks.
We chose seven datasets, two of which are non-domain-specific public datasets, to ensure the generality of our experiment.

We measure the performance of fine-tuned models by evaluating their ability to perform search tasks, as measured by Mean Reciprocal Rank (mRR), Recall, and Mean Average Precision (mAP).
These metrics are calculated using the top 20 results of each search in the validation subset held out from each dataset.

### How much labeled data is needed?

### How much time is needed?

## Summary