(retrieve-tuned-model)=
# Save Artifact

Perfect!
Now, you have started the fine-tuning job in the Jina AI Cloud.
When the fine-tuning job is finished, the resulting model is automatically stored under your Jina account in the Jina AI Cloud.
Next, we can get its artifact id and download the model.

```{admonition} Managing fine-tuned models
:class: hint
To use a fine-tuned model in a Jina service running on [JCloud](https://github.com/jina-ai/jcloud), you do not need to download it.
Each model has a artifact id, which is sufficient to setup an encoding serivce as explained in the section {doc}`/walkthrough/integrate-with-jina`.
Alternatively, you can also download the model using the artifact id, as explained below, e.g., to use it in a locally runnig Jina service. 
```

Please note that fine-tuning takes time. It highly depends on the size of your training data, evaluation data, and other hyperparameters.
Because of this, you might have to close the session and reconnect to it several times.

In the example below, we show how to connect to an existing run and download a tuned model:

```python
import finetuner

finetuner.login()

# connect to the run we created previously.
run = finetuner.get_run(
    run_name='finetune-flickr-dataset-efficientnet-1',
    experiment_name='finetune-flickr-dataset',
)
print(f'Run status: {run.status()}')
print(f'Run artifact id: {run.artifact_id}')
```

You can monitor your run status in two ways:

1. Log streaming: Pull logs from Jina AI Cloud lively, suitable for small fine-tuning tasks.
2. Query logs: Pull up-to-date logs from Jina AI Cloud, suitable for long-running tasks.

````{tab} Stream logs
```python
for entry in run.stream_logs():
    print(entry)
```
````
````{tab} Query logs
```python
print(run.status())
print(run.logs())
```
````

Once run status is `FINISHED`, you can save the artifact with:

```python
run.save_artifact('tuned_model')
```

```{admonition} Share artifact with others
:class: hint
Finetuner allows you to set your artifact as a public artifact.
At training time, you need to set `public=True` when calling the `fit` function.
If `public=True`, anyone who knows the artifact id can download your artifact with the above function.
```

If the fine-tuning is finished, you will see the following message in the terminal:

```bash
🔐 Successfully logged in to Jina AI as [USER NAME]!
Run status: FINISHED
Run Artifact id: 62972acb5de25a53fdbfcecc
Run logs:

  Training [2/2] ━━━━━━━━━━━━━━━━━━━━━━━━━━━ 50/50 0:00:00 0:01:08 • loss: 0.050
[09:13:23] INFO     [__main__] Done ✨                           __main__.py:214
           INFO     [__main__] Saving fine-tuned models ...      __main__.py:217
           INFO     [__main__] Saving model 'model' in           __main__.py:228
                    /usr/src/app/tuned-models/model ...                         
           INFO     [__main__] Pushing saved model to Hubble ... __main__.py:232
[09:13:54] INFO     [__main__] Pushed model artifact ID:         __main__.py:238
                    '62972acb5de25a53fdbfcecc'                                  
           INFO     [__main__] Finished 🚀                       __main__.py:240```
```
