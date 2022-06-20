# Welcome to Finetuner!

```{include} ../README.md
:start-after: <!-- start elevator-pitch -->
:end-before: <!-- end elevator-pitch -->
```

**Finetuner** is an open-source offering by [Jina AI](https://jina.ai/) ✨

It enables users to fine-tune large pre-trained deep learning models in their specific domains and datasets. It handles the
infrastructure and the complexity of the fine-tuning task and provides a simple interface to submit fine-tuning jobs on the Jina Cloud.

Finetuner primarily targets business users and engineers with limited knowledge in Machine Learning, but also attempts to expose
lots of configuration options for experienced professionals!

## Why do I need it?

Search quality matters. When you bring a pre-trained model to encode your data to embeddings, you are likely to get irrelevant search results.
Pre-trained deep learning models are usually trained on large-scale datasets, that have a different *data distribution* over your own datasets or domains.
This is referred to as a *distribution shift*.

Finetuner provides a solution to this problem by leveraging a pre-trained model from a large dataset and fine-tuning the parameters of
this model on your dataset.

Once fine-tuning is done, you get a model adapted to your domain. This new model leverages better search performance on your-task-of-interest.

Fine-tuning a pre-trained model includes a certain complexity and requires Machine Learning plus domain knowledge (on NLP, Computer Vision e.t.c).
Thus, it is a non-trivial task for business owners and engineers who lack the practical deep learning knowledge. Finetuner attempts
to address this by providing a simple interface, which can be as easy as:

```python
import finetuner
from docarray import DocumentArray

# Login to Jina ecosystem
finetuner.login()

# Prepare training data
train_data = DocumentArray(...)

# Fine-tune in the cloud
run = finetuner.fit(
    model='resnet50', train_data=train_data, epochs=5, batch_size=128,
)

print(run.name)
print(run.logs())

# When ready
run.save_model(path='tuned_model')
```

You should see this in your terminal:

```bash
🔐 Successfully login to Jina Ecosystem!
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

Submitted fine-tuning jobs run efficiently on the Jina Cloud on either CPU or GPU enabled hardware.

Finetuner fully owns the complexity of setting up and maintaining the model training infrastructure plus the complexity of delivering SOTA training
methods to production use cases.

```{Important}
Not sure which model to use?

Don't worry, call `finetuner.describe_models()` and we will help you choose the best fit.
```


```{include} ../README.md
:start-after: <!-- start support-pitch -->
:end-before: <!-- end support-pitch -->
```

```{toctree}
:caption: How it Works
:hidden:

1_how_it_works/1_1_how_it_works.md
1_how_it_works/1_2_difference.md
```

```{toctree}
:caption: Get Started
:hidden:

2_step_by_step/2_1_install.md
2_step_by_step/2_2_experiment_and_run.md
2_step_by_step/2_3_login_to_jina_ecosystem.md
2_step_by_step/2_4_create_training_data.md
2_step_by_step/2_5_choose_backbone.md
2_step_by_step/2_6_start_finetune.md
2_step_by_step/2_7_get_tuned_model.md
2_step_by_step/2_8_next_steps.md
```

```{toctree}
:caption: {octicon}`book` How-To
:hidden:

3_how_to/3_1_text_to_text.md
3_how_to/3_2_image_to_image.md
3_how_to/3_3_text_to_image.md
```

```{toctree}
:caption: Developer Reference
:hidden:
:maxdepth: 1

api/finetuner
```

---
{ref}`genindex` {ref}`modindex`
