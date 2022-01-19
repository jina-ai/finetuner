# Finetuning ResNet50 on Totally Looks Like Dataset with Self-Supervised Pre-training

In a real-world scenario, not every dataset has labels while fine-tuning.
And it is non-trivial to collect a large number of quality labels on customized data.
Luckily, Finetuner allows you to Finetune a pre-trained {term}`embedding_model` without full supervision.

This tutorial uses totally looks like dataset, but we'll pre-train an {term}`embedding model` without using labels.
This is usually referred to as **self-supervised pre-training**.
After this step, we utilize 10% of the dataset with labels to improve the search quality.

This tutorial is a follow-up of [Finetuning ResNet50 on Totally Looks Like Dataset](../totally-looks-like/index.md).
For dataset introduction or fully supervised model fine-tuning,
please refer to that tutorial.


## Preparing Training data

## Self-Supervised Pre-training

## Fine-tuning Given Limited Supervision

## Result Evaluation

## Wrapping Up