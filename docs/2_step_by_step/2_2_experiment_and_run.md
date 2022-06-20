(experiment-and-runs)=
# Two basic concepts: Experiment and Run

Finetuner organizes your training based on two concepts: {class}`~finetuner.experiment.Experiment` and {class}`~finetuner.run.Run`.

An Experiment defines the machine learning task you're fine-tuning for.
A Run is a piece of code that performs the Experiment with specific configurations.
An Experiment contains a list of Runs,
each with different configurations.
For example:

+ Experiment: Fine-tune a transformer on the QuoraQA dataset.
  - Run1: Use bert-based model.
  - Run2: Use sentence-transformer model.
+ Experiment: Fine-tune ResNet on WILD dataset.
  - Run1: Use ResNet18 with learning rate 0.01 and SGD optimizer.
  - Run2: Use ResNet50 with learning rate 0.01 and SGD optimizer.
  - Run3: Use ResNet50 with learning rate 0.0001 and Adam optimizer.

When you start the fine-tuning job, you can declare the `experiment_name` and `run_name` like this:

```python
import finetuner

finetuner.fit(
  ...,
  experiment_name='quora-qa-finetune',
  run_name='quora-qa-finetune-bert',
)
```

Please note that these two arguments are optional.
If not supplied,
Finetuner will use the current working directory as a default `experiment_name`,
and generate a random `run_name` for you.