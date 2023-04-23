(callbacks)=
# {octicon}`link` Callbacks

Callbacks are a way of adding additional methods to the finetuning process.
The methods are executed when certain events occur and there are several callback classes, each serving a different function by providing different methods for different events.
A run can be assigned multiple callbacks using the optional `callbacks` parameter when it is created.

## EvaluationCallback

The `EvaluationCallback` is used to calculate performance metrics for the model being tuned at the end of each epoch.
In order to evaluate the model, two additional data sets - a query dataset and an index dataset - need to be provided as arguments.
If no index set is provided, the query dataset is reused instead.
To use `EvaluationCallback`:

```python
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

The evaluation callback is triggered at the end of each epoch, in which the model is evaluated using the `query_data` and `index_data` datasets that were provided when the callback was created.
These datasets can be provided in the same way the `train_data` and `eval_data` parameters of the {meth}`~finetuner.fit` method; either as a path to a CSV file, a {class}`~finetuner.DocumentArray` or the name of a {class}`~finetuner.DocumentArray` that has been pushed on the Jina AI Cloud. See {doc}`/concepts/data-preparation/` for more information about how to prepare your data.

It is worth noting that the evaluation callback and the `eval_data` parameter of the fit method do not do the same thing.
The `eval_data` parameter is used to evaluate the loss of the model.
On the other hand, the evaluation callback is used to evaluate the quality of the searches using metrics such as average precision and recall.
These search metrics can be used by other callbacks if the evaluation callback is first in the list of callbacks when creating a run.

```{admonition} Evaluation callback with two models
:class: hint
Usually, you don't need to provide the name of a model to the evalution callback.
The callback just takes the model which is fine-tuned.
However, if multiple models are involved in the fine-tuning process, like this is the case for CLIP models, it needs to be clear which model is used to encode the documents in `query_data` and `index_data`.
This can be specified by the `model` attribute of the callback.
If a different model should be used for the `index_data`, you can set this via the `index_model` attribute.
For an example, see {doc}`/notebooks/text_to_image`.
```

```{admonition} Gather example results
:class: hint
In addition to evaluation metrics,
you may find it helpful to see actual query results.
We have introduced the parameter `gather_examples` to the `EvaluationCallback` to make this easy.
If this parameter is set to True, the evaluation callback also tracks the Top-K results for some example queries samples from the query dataset:

You can retrieve the query results, before and after fine-tuning, with the `run.display_examples()`.
```