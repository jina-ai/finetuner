(using-callbacks)=

# Using Callbacks

Callbacks are a way of adding additional methods to the finetuning process. The methods are executed when certain events occur and there are several callback classes, each serving a different function by providing different methods for different events.
A run can be assigned multiple callbacks using the optional `callbacks` parameter when it is created.

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

The evaluation callback is triggered at two stages: at the start of fitting to generate progress bars to track the evaluation progress, and at the end of each epoch, in which the model is evaluated using the query and index datasets that were provided when callback was created.

## Early Stopping

The early stopping callback measures a given metric at the end of every epoch, and will stop the fitting early if the metric doesnt increase at a high enough rate. The best performing model out of the completed epochs is stored in the case that it starts to perform worse after a certain point. Both the minimum improvement and the number of successive rounds without improvement before the training ends can be specified as optional arguments at construction.
The example output below shows a situation in which the early stopping callback is used to end a training run early. Compared to the results above in the evaluation explanation, the number of epochs was only two, compared to the full five, more than halfing the runtime of the training. On the other hand, the final loss recorded is 0.007, more than twice as high as if all five training epochs had completed. While the parameters of the early stopping callback can be adjusted to further improve either the runtime or loss, however this tradeoff between the two outcomes is what the early stopping callback promises to deliver.

```bash
  Training [2/5] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 76/76 0:00:00 0:00:16 • loss: 0.007
[14:34:56] INFO     Done ✨                                                                              __main__.py:194
           DEBUG    Finetuning took 0 days, 0 hours 1 minutes and 35 seconds                             __main__.py:196
           DEBUG    Metric: 'resnet50_average_precision' Value: 0.15114                                  __main__.py:205
           DEBUG    Metric: 'resnet50_dcg_at_k' Value: 0.21535                                           __main__.py:205
           DEBUG    Metric: 'resnet50_f1_score_at_k' Value: 0.03291                                      __main__.py:205
           DEBUG    Metric: 'resnet50_hit_at_k' Value: 0.34551                                           __main__.py:205
           DEBUG    Metric: 'resnet50_ndcg_at_k' Value: 0.21535                                          __main__.py:205
           DEBUG    Metric: 'resnet50_precision_at_k' Value: 0.01728                                     __main__.py:205
           DEBUG    Metric: 'resnet50_r_precision' Value: 0.15114                                        __main__.py:205
           DEBUG    Metric: 'resnet50_recall_at_k' Value: 0.34551                                        __main__.py:205
           DEBUG    Metric: 'resnet50_reciprocal_rank' Value: 0.15114                                    __main__.py:205
           INFO     Building the artifact ...                                                            __main__.py:207
           INFO     Pushing artifact to Hubble ...                                                       __main__.py:231
```

The early stopping callback is triggered at the end of both training and evaluation batches, to record the training and evaluation loss respectively. It is also called at the end of each epoch to first calculate the mean loss of the chosen metric, training or evalution loss, then determine if there has been any improvement on the best model of the previous epochs, and finally to end training if there has not been a significant improvement in a given number of rounds.
