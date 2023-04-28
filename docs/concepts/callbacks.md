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

The `query_data` and `index_data` datasets are specified in the same format as the `train_data` and `eval_data` parameters of the {meth}`~finetuner.fit` method;
either as a path to a CSV file, a {class}`~finetuner.DocumentArray` or the name of a {class}`~finetuner.DocumentArray` that has been pushed to the Jina AI Cloud.
See {doc}`/concepts/data-preparation/` for more information about preparing your data.

It is worth noting that the evaluation callback and the `eval_data` parameter of the fit method do not do the same thing.
The `eval_data` parameter is used to evaluate model loss during training,
while the evaluation callback is used to measure model quality at the end of each epoch using metrics such as average precision and recall.
Other callbacks may use these metrics if the evaluation callback is first in the list of callbacks when creating a run.
These search metrics can be used by other callbacks if the evaluation callback is first in the list of callbacks when creating a run.

```{admonition} Evaluation callback with two models
:class: hint
You don't usually need to provide the name of a model to the evaluation callback.
It assumes you mean to evaluate the model that you are fine-tuning.
However, if multiple models are involved in the fine-tuning process,
for example, if you fine-tune CLIP models,
then you need to be clear about which model to use to encode the documents in `query_data` and `index_data`.
This is specified by the `model` parameter of the callback.
If the `index_data` should be encoded by a different model from the `query_data`,
you must specify this with the `index_model` parameter.
For an example, see {doc}`/notebooks/text_to_image`.
```

### Show evaluation metrics

During the fine-tuning process, the evaluation metrics are displayed in the logs, which you can retrieve via the {func}`~finetuner.run.Run.logs()` function.
After the fine-tuning job has finished, the evaluation metrics can be retrieved from the cloud by calling the {func}`~finetuner.run.Run.metrics()` function.
This function returns a JSON object with the metrics before and after fine-tuning.
Alternatively, you can also retrieve the metrics via the {func}`~finetuner.run.Run.display_metrics()` function, which prints the evaluation results in the form of a table to the console.

![Evaluation Metrics](https://user-images.githubusercontent.com/6599259/224283786-5803fb8e-6d40-4eb7-b1d2-ae91a24648e7.png)

### Display example results

If you want to compare the K most similar retrieval results (Top-K) before and after fine-tuning,
you can set the `gather_examples` parameter of the evaluation callback to `True`.
In this case, the evaluation callback will store the top-k results for each query document before and after fine-tuning.
You can retrieve them with the {func}`~finetuner.run.Run.example_results()` function. 
Alternatively, you can use the {func}`~finetuner.run.Run.display_examples()` function to display a table of the Top-K results before and after fine-tuning to the console.

![Example Results](https://user-images.githubusercontent.com/6599259/224284912-f3f6f547-8d75-4529-8df1-7f9904423239.png)


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

Similarly to the best model checkpoint callback,
the early stopping callback measures a given metric at the end of every epoch.
Unlike the best model checkpoint callback,
the early stopping callback does not save the best model;
only the monitored metric is recorded between runs in order to assess the rate of improvement.

Below is some example output for a run with the early stopping callback followed by the output for the same run without the early stopping callback,
and then the python code used to create the run.
The output for the run with early stopping finished after just ten epochs whereas the other run finished all twenty epochs,
resulting in nearly twice the runtime.
That said, the resulting loss value of the early stopping run is only 0.284,
compared to the full run's 0.272, less than five percent higher.
The early stopping callback can be used in this way to reduce the amount of training time while still showing improvement.

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

The early stopping callback triggers at the end of training and evaluation batches to record the loss, and at the end of each epoch to evaluate the model and compare it to the best so far. Whether it stops training at the end of an epoch depends on several parameters:

- `minimum_delta`: The minimum amount of improvement that a model can have over the previous best model to be considered worthwhile, zero by default, meaning that the training will not stop early unless the performance starts to decrease
- `patience`: The number of consecutive rounds without improvement before the training is stopped, two by default.
- `baseline`: an optional parameter that is used to compare the model's score against instead of the best previous model when checking for improvement. This baseline does not get changed over the course of a run.

## WiSEFTCallback

WiSE-FT, proposed by Mitchell et al. in [Robust fine-tuning of zero-shot models](https://arxiv.org/abs/2109.01903),
has been proven to be an effective way for fine-tuning models with a strong zero-shot capability,
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
