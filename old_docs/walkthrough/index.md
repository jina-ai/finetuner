# {octicon}`list-ordered` Walkthrough

Why do I need Finetuner?

Because search quality matters. 

When you bring a pre-trained model to encode your data to embeddings, you are likely to get irrelevant search results.
Pre-trained deep learning models are usually trained on large-scale datasets, that have a different *data distribution* over your own datasets or domains.
This is referred to as a *distribution shift*.

Finetuner provides a solution to this problem by leveraging a pre-trained model from a large dataset and fine-tuning the parameters of
this model on your dataset.

Once fine-tuning is done, you get a model adapted to your domain. This new model leverages better search performance on your-task-of-interest.

Fine-tuning a pre-trained model includes a certain complexity and requires Machine Learning plus domain knowledge (on NLP, Computer Vision, etc.).
Thus, it is a non-trivial task for business owners and engineers who lack practical deep-learning knowledge. Finetuner attempts
to address this by providing a simple interface, which can be as easy as:

```python
import finetuner
from finetuner import DocumentArray

# Login to Jina AI Cloud
finetuner.login()

# Prepare training data
train_data = DocumentArray(...)

# Fine-tune in the cloud
run = finetuner.fit(
    model='resnet50', train_data=train_data, epochs=5, batch_size=128,
)

print(run.name)
for log_entry in run.stream_logs():
    print(log_entry)

# When ready
run.save_artifact(directory='experiment')
```

You should see this in your terminal:

```bash
🔐 Successfully logged in to Jina AI as [USER NAME]!
Run name: vigilant-tereshkova
Run logs:

  Training [2/2] ━━━━━━━━━━━━━━━━━━━━━━━━━━━ 50/50 0:00:00 0:01:08 • loss: 0.050
[09:13:23] INFO     [__main__] Done ✨                           __main__.py:214
           INFO     [__main__] Saving fine-tuned models ...      __main__.py:217
           INFO     [__main__] Saving model 'tuned_model' in     __main__.py:228
                    /usr/src/app/tuned-models/model ...                         
           INFO     [__main__] Pushing saved model to Hubble ... __main__.py:232
[09:13:54] INFO     [__main__] Pushed model artifact ID:         __main__.py:238
                    '62972acb5de25a53fdbfcecc'                                  
           INFO     [__main__] Finished 🚀                       __main__.py:240
```

Submitted fine-tuning jobs run efficiently on the Jina AI Cloud on either CPU or GPU enabled hardware.

Finetuner fully owns the complexity of setting up and maintaining the model training infrastructure plus the complexity of delivering SOTA training methods to production use cases.

Please check out the following steps for more information:


```{toctree}
basic-concepts
login
create-training-data
choose-backbone
run-job
save-model
inference
```