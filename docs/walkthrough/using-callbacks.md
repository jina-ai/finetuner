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

## Evaluation

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

The evaluation callback is triggered at two stages: at the start of fitting to generate progress bars to track the evaluation progress, and at the end of each epoch, in which the model is evaluated using the query and index datasets that were provided when callback was created. It is worth noting that the evaluation callback and the eval_data parameter do not do the same thing. The eval data parameter takes a Document Array (or the name of one that has been pushed on Hubble) and uses its contents to evaluate the loss pf the model whereas the evaluation callback is used to evaluate the quality of the searches using metrics such as average precision and recall. These search metrics can be used by other callbacks if the evaluation callback is first in the list of callbacks when creating a run.

## Best Model Checkpoint

This callback evaluated the performance of the model at the end of each epoch, and keeps a record of the best perfoming model across all epochs. Once fitting is finished the best performing model is saved. The definition of 'best' used by the callback is provided by the `monitor` parameter of the callback. By default this value is 'val_loss', the loss function calculated using the evaluation data, however the loss calculated on the training data can be used instead with 'train_loss'; any metric that is recorded by the evaluation callback can also be used. Then, the `mode` parameter specifies wether the monitored metric should be maximised ('max') or minimised ('min'). By default the mode is set to auto, meaning that it will automatically chose the correct mode depending on the chosen metric: 'min' if the metric is loss and 'max' if the metric is one recorded by the evaluation callback.

EXAMPLE

This callback is triggered at the end of both training and evaluation batches, to record the losses of the two data sets, and is triggered a third time at the end of each epoch to evaluate the performance of the current model using the monitored metric and then record this model if it performs better than the best model so far.

## Early Stopping

Similarly to the best model checkpoint callback, the early stopping callback measures a given metric at the end of every epoch and saves the best performing model at the end of the fitting process. Unlike the best model checkpoint callback, the early stopping callback will end the fitting process early if the metric does not improve enough between successive runs. Since the best model is only used to assess the rate of improvement, only the monitored metric is needed and so the model itself is not saved.

REDO THIS PART FOR NEW OUTPUT
The examples below show some code in which the early stopping callback is used, along with the output logs at the end of the tuning. Compared to the results in previous explanations only , the number of epochs was only, compared to the full five, more than halfing the runtime of the training. On the other hand, the final loss recorded is 0.007, more than twice as high as if all five training epochs had completed. While the parameters of the early stopping callback can be adjusted to further improve either the runtime or loss, however this tradeoff between the two outcomes is what the early stopping callback promises to deliver.

```python
run = finetuner.fit(
    model = 'resnet50',
    run_name = 'resnet-tll-early-callback',
    train_data = 'tll-train-da',
    batch_size = 128,
    epochs = 5,
    learning_rate = 1e-6,
    cpu = False,
    callbacks=[
        callback.EarlyStopping(
            monitor = "train_loss",
            mode = "min",
            patience=3, 
            min_delta=0.0001,
            baseline = 0.006
            )
    ]
)
```

```bash
  Training [2/5] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 76/76 0:00:00 0:00:16 • loss: 0.007
[14:34:56] INFO     Done ✨                                                                              __main__.py:194
           INFO     Building the artifact ...                                                            __main__.py:207
           INFO     Pushing artifact to Hubble ...                                                       __main__.py:231
```

The early stopping callback triggers at the same times as the best model checkpoint callback: at the end of training and evaluation batches to record the loss, and at the end of each epoch to evaluate the model and compare it to the best so far. It differs from the best model checkpoint after this. If the best model is not improved upon by a certain amount specified by the `minimum_delta` parameter, then it is not counted as having improved that epoch (the best model is still updated). If the model does not show improvement after a number of rounds specified by the `patience` parameter, then training is ended early. By default, the `minimum_delta` parameter is zero and the `patience` parameter is two.
The early stopping callback is triggered at the end of both training and evaluation batches, to record the training and evaluation loss respectively.

## Training Checkpoint

The training checkpoint saves a copy of the tuner at the end of each epoch, or the last k epochs, as determined by the `last_k_epochs` parameter which is one by default.

DOT DOT DOT

The training checkpoint callback is only triggered at the end of each epoch. In this time it saves the tuner in its current state and appends it to a list of checkpoints. If the list already has a length of k, then the oldest state is removed from the list.
