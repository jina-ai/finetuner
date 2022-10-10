(using-callbacks)=
# Using Callbacks

Callbacks are a way of adding additional methods to the finetuning process. The methods are executed when certain events occur and there are several callback classes, each serving a different function by providing different methods for different events.
A run can be assigned multiple callbacks using the optional `callbacks` parameter when it is created.

```python
run = finetuner.fit(
    model = 'resnet50',
    run_name = 'resnet-tll-early-6',
    train_data = 'tll-train-da',
    epochs = 5,
    learning_rate = 1e-6,
    callbacks=[
        EvaluationCallback(
            query_data='tll-test-query-da',
            index_data='tll-test-index-da'
        ),
        EarlyStopping()
    ]
)
```

## EvaluationCallback

The evaluation callback is used to calculate performance metrics for the model being tuned at the end of each epoch. In order to evaluate the model, two additional data sets - a query dataset and an index dataset - need to be provided as arguments. If no index set is provided, the query dataset is reused instead. Below is an example of the metrics as they are output at the end of finetuning:

```bash
  Training [5/5] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 76/76 0:00:00 0:00:16 • loss: 0.003
[14:10:40] INFO     Done ✨                                                                              __main__.py:194
           DEBUG    Finetuning took 0 days, 0 hours 3 minutes and 38 seconds                             __main__.py:196
           DEBUG    Metric: 'resnet50_average_precision' Value: 0.16654                                  __main__.py:205
           DEBUG    Metric: 'resnet50_dcg_at_k' Value: 0.24018                                           __main__.py:205
           DEBUG    Metric: 'resnet50_f1_score_at_k' Value: 0.03631                                      __main__.py:205
           DEBUG    Metric: 'resnet50_hit_at_k' Value: 0.38123                                           __main__.py:205
           DEBUG    Metric: 'resnet50_ndcg_at_k' Value: 0.24018                                          __main__.py:205
           DEBUG    Metric: 'resnet50_precision_at_k' Value: 0.01906                                     __main__.py:205
           DEBUG    Metric: 'resnet50_r_precision' Value: 0.16654                                        __main__.py:205
           DEBUG    Metric: 'resnet50_recall_at_k' Value: 0.38123                                        __main__.py:205
           DEBUG    Metric: 'resnet50_reciprocal_rank' Value: 0.16654                                    __main__.py:205
           INFO     Building the artifact ...                                                            __main__.py:207
           INFO     Pushing artifact to Hubble ...                                                       __main__.py:231
```

The evaluation callback is triggered at the end of each epoch, in which the model is evaluated using the `query_data` and `index_data` datasets that were provided when the callback was created.
It is worth noting that the evaluation callback and the `eval_data` parameter of the fit method do not do the same thing. The eval data parameter takes a `DocumentArray` (or the name of one that has been pushed on the cloud) and uses its contents to evaluate the loss of the model whereas the evaluation callback is used to evaluate the quality of the searches using metrics such as average precision and recall. These search metrics can be used by other callbacks if the evaluation callback is first in the list of callbacks when creating a run.

## BestModelCheckpoint

This callback evaluates the performance of the model at the end of each epoch, and keeps a record of the best perfoming model across all epochs. Once fitting is finished the best performing model is saved instead of the most recent model. The definition of best is based on two parameters:

- `monitor`: The metric that is used to compare models to each other. By default this value is `val_loss`, the loss function calculated using the evaluation data, however the loss calculated on the training data can be used instead with `train_loss`; any metric that is recorded by the evaluation callback can also be used.
- `mode`: Whether the monitored metric should be maximised (`max`) or minimised (`min`). By default the mode is set to `auto`, meaning that it will automatically choose the correct mode depending on the chosen metric: 'min' if the metric is loss and 'max' if the metric is one recorded by the evaluation callback.

The console output below shows how the evaluation loss of the model is monitored between each epoch, and how the best performing model is tracked. Since the final model has a higher loss than the previously recorded best model, the best model will be saved instead of the latest one.

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

Similarly to the best model checkpoint callback, the early stopping callback measures a given metric at the end of every epoch. Unlike the best model checkpoint callback, the early stopping callback does not save the best model; only the monitored metric is recorded between runs in order to assess the rate of improvement.

Below is some example output for a run with the early stopping callback followed by the output for the same run without the early stopping callback, and then the python code used to create the run. The output for the run with early stopping finished after just ten epochs whereas the other run finished all twenty epochs, resulting in nearly twice the runtime. That said, the resulting loss value of the early stopping run is only 0.284, compared to the full run's 0.272, less than five percent higher. The early stopping callback can be used in this way to reduce the amount of training time while still showing improvement.

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
from finetuner.callback import EarlyStopping, EvaluationCallback

run = finetuner.fit(
    model='openai/clip-vit-base-patch32',
    run_name='clip-fashion-early',
    train_data='clip-fashion-train-data',
    epochs=10,
    learning_rate= 1e-5,
    loss='CLIPLoss',
    cpu=False,
    callbacks= [
        callback.EarlyStopping(
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
