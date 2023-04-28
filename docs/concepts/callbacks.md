(callbacks)=
# {octicon}`link` Callbacks

A callback is a function provided as an argument to another function that can optionally run when a specific kind of event occurs.
There are several events during a Finetuner run that support callbacks. 
You can assign callbacks to the `finetuner.fit` method with the optional `callbacks` parameter.

## EvaluationCallback

The `EvaluationCallback` calculates retrieval metrics at the end of each epoch for the model being tuned.
In order to evaluate the model, two additional data sets - a query dataset and an index dataset - need to be provided as arguments.
If no index set is provided, the evaluation is performed with the query dataset.
To use `EvaluationCallback`:

```python
import finetuner
from finetuner.callback import EvaluationCallback

run = finetuner.fit(
    ...,
    callbacks=[
        EvaluationCallback(
            query_data='finetuner/tll-test-query-da',
            index_data='finetuner/tll-test-index-da'
        ),
    ]
)
```

Below is an example of the metrics as they are printed at the end of fine-tuning:

```bash
Training [5/5] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 48/48 0:00:00 0:00:12 • loss: 13.986
INFO     Done ✨                                                                                          __main__.py:195
DEBUG    Finetuning took 0 days, 0 hours 3 minutes and 48 seconds                                         __main__.py:197
INFO     Metric: 'resnet_base_precision_at_k' before fine-tuning:  0.11575 after fine-tuning: 0.53425     __main__.py:210
INFO     Metric: 'resnet_base_recall_at_k' before fine-tuning:  0.05745 after fine-tuning: 0.27113        __main__.py:210
INFO     Metric: 'resnet_base_f1_score_at_k' before fine-tuning:  0.07631 after fine-tuning: 0.35788      __main__.py:210
INFO     Metric: 'resnet_base_hit_at_k' before fine-tuning:  0.82900 after fine-tuning: 0.94100           __main__.py:210
INFO     Metric: 'resnet_base_average_precision' before fine-tuning:  0.52305 after fine-tuning: 0.79779  __main__.py:210
INFO     Metric: 'resnet_base_reciprocal_rank' before fine-tuning:  0.64909 after fine-tuning: 0.89224    __main__.py:210
INFO     Metric: 'resnet_base_dcg_at_k' before fine-tuning:  1.30710 after fine-tuning: 4.52143           __main__.py:210
INFO     Building the artifact ...                                                                        __main__.py:215
INFO     Pushing artifact to Jina AI Cloud ...                                                            __main__.py:241
INFO     Artifact pushed under ID ''                                                                      __main__.py:243
DEBUG    Artifact size is 83.580 MB                                                                       __main__.py:245
INFO     Finished 🚀    
```

For a more detailed explanation of parameters and usage, please refer to the {ref}`evaluation <evaluation>` page.


## BestModelCheckpoint

This callback evaluates the performance of the model at the end of each epoch, and keeps a record of the best performing model across all epochs.
Once fitting is finished the best performing model is saved instead of the most recent model.
Finetuner determines which model is the best is based on two parameters:

- `monitor`: This parameter is by default `val_loss`, which uses the evaluation data to compare models. Alternatively, you can set this to `train_loss`, which will compare models using the training data. You can specify any metric recorded by the evaluation callback for this parameter.
- `mode`: Whether the monitored metric should be maximised (`max`) or minimised (`min`). By default the mode is set to `auto`, meaning that it will automatically choose the correct mode depending on the chosen metric: 'min' if the metric is loss and 'max' if the metric is one recorded by the evaluation callback.

The console output below shows how the evaluation loss of the model is monitored between epochs and how the best-performing model is tracked. Since the final model has a higher loss than the previously recorded best model, the best model will be saved instead of the latest one.

```bash
           INFO     Finetuning ...                                                   
                    __main__.py:173
[11:50:33] INFO     Model improved from inf to 2.756!                                
       best_model_checkpoint.py:112
[11:50:52] INFO     Model improved from 2.756 to 2.711!                              
       best_model_checkpoint.py:112
[11:51:10] INFO     Model did not improve                                            
       best_model_checkpoint.py:120
[11:51:28] INFO     Model did not improve                                            
       best_model_checkpoint.py:120
  Training [4/4] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 54/54 0:00:00 0:00:15 • loss: 0.496 • val_loss: 2.797
           INFO     Done ✨                                                          
                    __main__.py:194
           DEBUG    Finetuning took 0 days, 0 hours 1 minutes and 16 seconds         
                    __main__.py:196
           INFO     Building the artifact ...                                        
                    __main__.py:207
           INFO     Pushing artifact to Hubble ...                                   
                    __main__.py:231
```

## EarlyStopping

The `EarlyStopping` callback runs at the end of each training epoch and decides,
based on user-provided criteria, if it should stop training and save the current model.

It takes `monitor` and `mode` parameters that work the same way they do for `BestModelCheckpoint` above.
However, instead of using this information to identify the best-performing model out of all epochs, it uses user-specified parameters at the end of each epoch to determine if it should terminate fine-tuning and save the current model.

Below is the output of a Finetuner run with early stopping, compared to one with the same data parameters but without early stopping.
Note that the early stopping run terminated after ten epochs, while the other run continued for the whole twenty epochs. The loss of the final model is, in the two cases, nearly identical.
The early stopping run has a loss of 0.284, while taking the whole 20 epochs produced a model with a loss of 0.272.
Early stopping halved the runtime at a cost to performance of less than five percent.

```bash
  Training [10/20] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 54/54 0:00:00 0:00:14 • loss: 0.284
[11:19:28] INFO     Done ✨                                                                              __main__.py:194
           DEBUG    Finetuning took 0 days, 0 hours 2 minutes and 30 seconds                             __main__.py:196
           INFO     Building the artifact ...                                                            __main__.py:207
           INFO     Pushing artifact to Hubble ...                                                       __main__.py:231
```

```bash
  Training [20/20] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 54/54 0:00:00 0:00:14 • loss: 0.272
[10:37:33] INFO     Done ✨                                                                              __main__.py:194
           DEBUG    Finetuning took 0 days, 0 hours 4 minutes and 54 seconds                             __main__.py:196
           INFO     Building the artifact ...                                                            __main__.py:207
           INFO     Pushing artifact to Hubble ...                                                       __main__.py:231
```

The user-specified parameters are:

- `min_delta`: The minimum amount of improvement that a model must have over the previous best model in order for fine-tuning to continue. By default, this is zero, meaning that the training will not stop early unless the performance starts to decrease.
- `patience`: The number of consecutive rounds without improvement before the training is stopped, set to two by default.
- `baseline`: an optional parameter that is used to compare the model's score against instead of the best previous model when checking for improvement. If specified, the improvement every epoch is measured by comparison to this value instead of the best-performing model so far.

In code:

```python
import finetuner
from finetuner.callback import EarlyStopping

run = finetuner.fit(
    ...,
    callbacks= [
        EarlyStopping(
            monitor = "train_loss",
            mode = "min",
            patience=2,
            min_delta=1,
            baseline=1.5
        )
    ]
)
```

## WiSEFTCallback

WiSE-FT, proposed by Mitchell et al. in [Robust fine-tuning of zero-shot models](https://arxiv.org/abs/2109.01903),
has proven to be an effective way of fine-tuning models with strong zero-shot capability,
such as CLIP:

> Large pre-trained models such as CLIP or ALIGN offer consistent accuracy across a range of data distributions when performing zero-shot inference (i.e., without fine-tuning on a specific dataset). Although existing fine-tuning methods substantially improve accuracy on a given target distribution, they often reduce robustness to distribution shifts. We address this tension by introducing a simple and effective method for improving robustness while fine-tuning: ensembling the weights of the zero-shot and fine-tuned models (WiSE-FT).

```python
import finetuner
from finetuner.callback import WiSEFTCallback

run = finetuner.fit(
    model='clip-base-en',
    ...,
    loss='CLIPLoss',
    callbacks=[WiSEFTCallback(alpha=0.5)]
)
```

Alpha should be assigned a value that is greater than or equal to 0 but less than or equal to 1:

1. When alpha is set to 0, the fine-tuned model is the same as the pre-trained model.
2. When alpha is set to 1, the pre-trained weights are not used.
3. When alpha is a floating-point number between 0 and 1, we combine the weights of the pre-trained and fine-tuned models.

```{warning}
It is recommended to use WiSEFTCallback when fine-tuning CLIP.
We can not ensure it works for other types of models, such as ResNet or BERT.

Please refer to {ref}`Apply WiSE-FT <wise-ft>` in the CLIP fine-tuning example for more information.
```

## WandBLogger

Finetuner allows you to utilize Weights & Biases for experiment tracking and visualization.
The `WandBLogger` uses Weights & Biases [Anonymous Mode](https://docs.wandb.ai/ref/app/features/anon) to track a Finetuner Run.

```{admonition} Use WandBLogger together with EvaluationCallback
:class: hint
The WandBLogger will track the training loss, plus the evaluation loss if `eval_data` is not None.

If you use EvaluationCallback together with WandBLogger, search metrics will be tracked as well.
Such as `mrr`, `precision`, `recall`, etc.
```

```python
import finetuner
from finetuner.callback import WandBLogger, EvaluationCallback

run = finetuner.fit(
    ...,
    callbacks=[
        WandBLogger(),
        EvaluationCallback(
            query_data='finetuner/tll-test-query-da',
            index_data='finetuner/tll-test-index-da'
        ),
    ]
)

# stream the logs, or use run.logs() to get logs
for entry in run.stream_logs():
    print(entry)
```

You can find the Weights & Biases entry in the logs, copy the link of *View run at*:

```bash
wandb: Currently logged in as: anony-mouse-279369. Use `wandb login --relogin` to force relogin
wandb: wandb version 0.13.5 is available!  To upgrade, please run:
wandb:  $ pip install wandb --upgrade
wandb: Tracking run with wandb version 0.12.19
wandb: Run data is saved locally in [YOUR-PATH]
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run cool-wildflower-2
wandb:  View project at https://wandb.ai/anony-mouse-279369/[YOUR-PROJECT-URL]
wandb:  View run at https://wandb.ai/anony-mouse-279369/[YOUR-RUN-URL]
```

WandBLogger             |
:-------------------------:|
![wandblogger](https://user-images.githubusercontent.com/6599259/212645881-20071aba-8643-4878-bc53-97eb6f766bf0.png) | 
