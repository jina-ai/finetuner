(manage-runs)=
# {octicon}`terminal` Managing Runs

After starting a fine-tuning or a synthesis job by calling {meth}`~finetuner.fit` or {meth}`~finetuner.synthesize`,
those methods return a {class}`~finetuner.run.Run` object. This object holds information about your job and serves as a handle to access the job running on the Jina AI Cloud.

For example, the name of your run is in the `name` attribute of the `Run` object:

```python
run = finetuner.fit(
    ...
)

print(run.name)

```python
import finetuner

finetuner.login()

run = finetuner.get_run('your-run-name')
```
Your runs are only accessible to you, so you need to log in first.

If you used the `experiment_name` parameter in your call to {meth}`~finetuner.fit` or {meth}`~finetuner.synthesize` to specify an experiment name, 
you will need to add that to the `get_run` call:

```python
run = finetuner.get_run('your-run-name', 'your-experiment-name')
Further, you can use the {meth}`finetuner.list_runs` function to obtain a list of run objects for all jobs you submitted.

```{admonition} Run-time of the list_runs function
:class: hint
If this list is very long, the function call might take a while since it needs to send multiple requests.
```


## Attributes of a run

A `Run` object has the following attributes:
- `name`: It references a job in the Jina AI Cloud. Therefore, it has to be unique for its experiment. If you don't specify a name yourself, Finetuner will give it a memorable name.
- `config`: A configuration which is serialized from the attributes, which you passed to the `fit` or `synthesize` function.
- `artifact_id`: The results of your job are stored in an artifact on Jina AI Cloud identified by this artifact id. You can use this id in the {meth}`~finetuner.get_model` function to retrieve your fine-tuned mode, as explained in the [Inference Section](inference).

## Check the status of a run
You can check the status of a `Run` via the {meth}`~finetuner.run.Run.status` method:

```python
print(f'Run status: {run.status()}')
```

If you just launched the job, you'll likely see something like this in the terminal:

```bash
Run status: CREATED
```

During fine-tuning or synthesis, the run status changes from:
1. `CREATED`: The {class}`~finetuner.run.Run` has been created and submitted to the job queue.
2. `STARTED`: The job is in progress.
3. `FINISHED`: The job finished successfully and the model has been sent to Jina AI Cloud.
4. `FAILED`: The job failed, please check the logs for more details.

## Retrieve the logs of a run

If the status of your run is `STARTED` or `FINISHED`,
you can retrieve the job's logs from Jina AI Cloud with the {meth}`~finetuner.run.Run.logs` method.
This will return a list of all log messages produced by the job so far.

You can stream the logs to your console with the {meth}`~finetuner.run.Run.stream_logs` method.
If your run's status is `CREATED`, this method will wait until your job status is `STARTED` and then continuously prints logs to your console and continues until your job has either finished or failed

```python
for entry in run.stream_logs():
    print(entry)
```

In addition to log messages, you can obtain more tracking information with the `WandBLogger` callback.
See the [Callbacks Section](callbacks) for more details.

## View runs in Jina AI Cloud

You can also view your runs in the Jina AI Cloud.
After you login (see [Login Section](login)),
you can click on the "Finetuner" section in the left sidebar at the [cloud.jina.ai](https://cloud.jina.ai/) Website.
This will show you a list of all your runs:

![Jina AI Cloud - List of Runs](https://user-images.githubusercontent.com/6599259/233099591-d27405b3-a26c-4951-81df-2c5dc096113e.png)

To see more details about a specific run, like the logs and job configuration, click on its row:

![Jina AI Cloud - View Logs](https://user-images.githubusercontent.com/6599259/233099603-6af406e1-15c1-401b-af5a-495404114f4c.png)

## Saving artifacts

When a fine-tuning job finishes, it stores a model or models, and for synthesis jobs, it stores a new training dataset.
In either case, you can save this result with the {meth}`~finetuner.run.Run.save_artifact` method:

```python
run.save_artifact('path/for/tuned_model')
```

The [Inference Section](inference) describes how to use downloaded models. 

## Evaluation data

The `Run` object also has the methods {meth}`~finetuner.run.Run.metrics` and
{meth}`~finetuner.run.Run.example_results`, which retrieve results produced by an `EvaluationCallback`.
Details are provided in the [Evaluation Section](evaluation).