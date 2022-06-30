(retrieve-tuned-model)=
# Save Model

Perfect!
Now you have started the fine-tuning job in the cloud.
However, fine-tuning takes time. It highly depends on the size of your training data, evaluation data and other hyper-parameters. Because of this, you might have to close the session and reconnect to it several times.

In the following example, we show how to connect to an existing run and download a tuned model:

```python
import finetuner

finetuner.login()

# connect to the experiment we created previously.
experiment = finetuner.get_experiment('finetune-flickr-dataset')
# connect to the run we created previously.
run = experiment.get_run('finetune-flickr-dataset-efficientnet-1')
print(f'Run status: {run.status()}')
print(f'Run logs: {run.logs()}')
# save the model.
run.save_artifact('tuned_model')
```

If the fine-tuning finished,
you can see this in the terminal:

```bash
🔐 Successfully login to Jina Ecosystem!
Run status: FINISHED
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
