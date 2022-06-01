# Two basic concepts: Experiment and Run

Finetuner organize your training based on two concepts: **Experiment** and **Run**.

An Experiment is defined as a general purpose machine learning task you're fine-tuning for.
A Run is a piece of code that performs the Experiment with specific configurations.
An Experiment contains a list of Runs,
each with different configurations.
For example:

+ Experiment: Fine-tune transformer on QuoraQA dataset.
  - Run1: Use bert-based model
  - Run2: Use setence-transformer model.
+ Experiment: Fine-tune ResNet on WILD dataset.
  - Run1: Use ResNet18 with learning rate 0.01.
  - Run2: Use ResNet50 with learning rate 0.01.
  - Run3: Use ResNet50 with learning rate 0.0001.
+ Experiment: Fine-tune CLIP on H&M Fashion dataset.
  - Run1: Use batch size of 128 and SGD optimizer.
  - Run2: Use batch size of 256 and SGD optimizer.
  - Run3: Use batch size of 256 and Adam optimizer.
