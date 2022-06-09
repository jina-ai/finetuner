# Finetuner in Action Template
This is a template for the documentation guides of Finetuner in action, with the general structure and layout to be used for demonstrating how Finetuner can be deployed for different tasks.

```{admonition} See Also: Jina Contribution Guidelines
:class: seealso
For more info on best practices for documentation, see Jina's [contribution guidelines](https://github.com/jina-ai/jina/blob/master/CONTRIBUTING.md#-contributing-documentation)
```

## Task overview
Describe the task which this guide accomplishes, including which model will be fine-tuned and which dataset you will use for.

Also provide a brief description of what the task entails, what the dataset looks like and a high-level description of how the dataset is processed.


## Preparing data
Outline where the data can be found, artifact names in hubble or if relevant, how a user might load their own custom data. 
Add a link to supplementary dataset info, for example as a `See Also` {admonition}.
If you are outlining how to preprocess a dataset from scratch, use {dropdown} to hide long code snippets.


## Login to Finetuner
Explain how the user can login to Finetuner and why this is necessary. Feel free to also link to other documentation as well.

Example:

"As explained in the [Login to Jina ecosystem](../2_step_by_step/2_3_login_to_jina_ecosystem.md) section, first we need to login to Finetuner:"
```python
import finetuner
finetuner.login()
```


## Choosing the model
Always show the available models in your guide. Then mention which model will be used in your fine-tuning task. 
Feel free to add a `See Also` {admonition} for supplementary info on the model, perhaps a relevant paper or site.

Example:

"You can see all available models either in [the docs](../2_step_by_step/2_5_choose_back_bone.md) or by calling:"
```python
finetuner.describe_models()
```

```bash
                                                                  Finetuner backbones                                                                   
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                            model ┃           task ┃ output_dim ┃ architecture ┃                                          description ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│                                         resnet50 │ image-to-image │       2048 │          CNN │                               Pretrained on ImageNet │
│                                        resnet152 │ image-to-image │       2048 │          CNN │                               Pretrained on ImageNet │
│                                  efficientnet_b0 │ image-to-image │       1280 │          CNN │                               Pretrained on ImageNet │
│                                  efficientnet_b4 │ image-to-image │       1280 │          CNN │                               Pretrained on ImageNet │
│                     openai/clip-vit-base-patch32 │  text-to-image │        768 │  transformer │ Pretrained on millions of text image pairs by OpenAI │
│                                  bert-base-cased │   text-to-text │        768 │  transformer │       Pretrained on BookCorpus and English Wikipedia │
│ sentence-transformers/msmarco-distilbert-base-v3 │   text-to-text │        768 │  transformer │           Pretrained on Bert, fine-tuned on MS Marco │
└──────────────────────────────────────────────────┴────────────────┴────────────┴──────────────┴──────────────────────────────────────────────────────┘
```


## Creating a fine-tuning job
Show the user how to create a fine-tuning run, then explain why your example run has particular parameters and what they do. Also mention which parameters are optional or required.
Provide a more detailed explanation of parameters that are important for your particular experiment. 

Example:

```python
run = finetuner.fit(
    ...
    )
```
"Let's understand what this piece of code does ..."


Also show the user how they can monitor their run. Example:

"Now that we've created a run, let's see its status."
```python
print(run.status())
```

```bash
{'status': 'CREATED', 'details': 'Run submitted and awaits execution'}
```


## Reconnect and retrieve the runs
Show the user how to connect to their run, look at their logs and and save their model when fine-tuning has completed.

"Since some runs might take up to several hours/days, it's important to know how to reconnect to Finetuner and retrieve your run."
```python
import finetuner
finetuner.login()
run = finetuner.get_run('my_run')
```

"You can monitor the run by checking the status - `run.status()` or the logs - `run.logs()`. 

If your run has finished successfully, you can save fine-tuned models in the following way:"
```python
run.save_model('clip-model')
```

## Evaluation and performance
Explain to the user how they can track the performance of the model(s) they have fine-tuned in their runs. If this is not implemented yet, show the user an example log and how they might deduce model performance from this log.