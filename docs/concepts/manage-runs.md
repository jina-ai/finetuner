(manage-runs)=
# {octicon}`terminal` Managing Runs

After you have started a fine-tuning or a synthesis job by calling {meth}`~finetuner.fit` or {meth}`~finetuner.synthesize`, you get a {class}`~finetuner.run.Run` object in response.
This run object serves as a handle for the cloud job which you triggered.
If you lost the run object for the job, you can create a new `Run` object with the {meth}`finetuner.get_run` function:

```python
import finetuner

finetuner.login()

run = finetuner.get_run('your-run-name', 'your-experiment-name')
```
Since your runs are only accessible by you, you need to log in first.
If you did not specify an experiment name, when calling the `fit` or `synthesize` function, you can omit the experiment name, in the `get_run` function as well.
Further, you can use the {meth}`finetuner.list_runs` function to obtain a list of run objects for all jobs you submitted.

```{admonition} Run-time of the list_runs function
:class: hint
If this list is very long, the function call might take a while since it needs to send multiple requests.
```


## Attributes of a run

A `Run` object has the following attributes:
- `name`: It is used to identify a job in the Jina AI Cloud. Therefore, it has to be unique for its experiment. If you don't specify a name yourself, Finetuner will give it a memorable name.
- `config`: A configuration which is serialized from the attributed, which you passed to the `fit` or `synthesize` function.
- `artifact_id`: The outcome of your job is stored in an artifact on Jina AI Cloud. This attribute holds its id. You can use this artifact id for instance in the {meth}`~finetuner.get_model` function to retrieve the model for inference as explained in the [Inference Section](inference).

## Check the status of a run
You can check the status of a `Run` via the {meth}`~finetuner.run.Run.status` method:

```python
print(f'Run status: {run.status()}')
```

You'll see something like this in the terminal:

```bash
Run status: CREATED
```

During fine-tuning or synthesis
the run status changes from:
1. CREATED: the {class}`~finetuner.run.Run` has been created and submitted to the job queue.
2. STARTED: the job is in progress
3. FINISHED: the job finished successfully, model has been sent to Jina AI Cloud.
4. FAILED: the job failed, please check the logs for more details.


## Retrieve the logs of a run

When the status of your run is STARTED or FINISHED, you can retrieve logs from the cloud job with the {meth}`~finetuner.run.Run.logs` method.
This will return a list of all log messages, which were produced by the job so far.

Alternatively, you can streamline the logs to your console.
Therefore, Finetuner provides the {meth}`~finetuner.run.Run.stream_logs` method.
Executing the following code will wait until your job obtains the status STARTED 
if it is still CREATED and continuously prints logs to your console when they come in until the job is finished:

```python
for entry in run.stream_logs():
    print(entry)
```


## View runs in Jina AI Cloud

You can also view your runs in the Jina AI Cloud.
After you logged in (see [Login Section](login)), you can click on the "Finetuner" section in the left sidebar at the [cloud.jina.ai](https://cloud.jina.ai/) Website.
This will show you a list of all your runs:

![Jina AI Cloud - List of Runs](https://user-images.githubusercontent.com/6599259/233099591-d27405b3-a26c-4951-81df-2c5dc096113e.png)

To see more details of a specific run by clicking on the respective row.
This will show you more information, like the logs and the configuration of the job:

![Jina AI Cloud - View Logs](https://user-images.githubusercontent.com/6599259/233099603-6af406e1-15c1-401b-af5a-495404114f4c.png)

## Evaluation data

Also worth to mention is that a `Run` object has methods like {meth}`~finetuner.run.Run.metrics` and {meth}`~finetuner.run.Run.example_results` to retrieve results produced by an `EvaluationCallback`.
Those are described in detail in the [Evaluation Section](evaluation).