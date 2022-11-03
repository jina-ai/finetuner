# Finetuner in Action Template
This is a template for the documentation guides of Finetuner in action, with the general structure and layout to be used for demonstrating how Finetuner can be applied to solve different tasks.

```{admonition} See Also: Jina Contribution Guidelines
:class: seealso
For more info on best practices for documentation, see Jina's [contribution guidelines](https://github.com/jina-ai/jina/blob/master/CONTRIBUTING.md#-contributing-documentation)
```

## Task overview
Describe the task which this guide accomplishes, including which model will be fine-tuned and which dataset you will use.

Also provide a brief description of what the task entails, what the dataset looks like and a high-level description of how the dataset is processed.


## Preparing data
Outline where the data can be found, artifact names in Jina AI Cloud or if relevant, how a user might load their own custom data. 
Add a link to supplementary dataset info, for example as a `See Also` {admonition}.
If you are outlining how to preprocess a dataset from scratch, use {dropdown} to hide long code snippets.


## Choosing the model
Mention which model will be used in your fine-tuning task. Feel free to add a `See Also` {admonition} for supplementary info on the model, perhaps a relevant paper or site.

You can also add a `Tip` {admonition} for how the user can view all available models, also referring to the `Choose backbone model` documentation.


## Creating a fine-tuning run
Show the user how to create a fine-tuning run, then explain why your example run has particular parameters and what they do. Also mention which parameters are optional or required.
Provide a more detailed explanation of parameters that are important for your particular experiment. 

Example:

```python
run = finetuner.fit(
    ...
    )
```
"Let's understand what this piece of code does ..."


## Monitoring your runs

Also show the user how they can monitor their run, and reconnect to it if they were disconnected. 

Example:

"Now that we've created a run, let's see its status. You can monitor the run by checking the status - `run.status()` or the logs - `run.logs()`. "
```python
print(run.status())
```

```bash
{'status': 'CREATED', 'details': 'Run submitted and awaits execution'}
```

"Since some runs might take up to several hours/days, you can reconnect to your run very easily to monitor its status and logs."
```python
import finetuner
finetuner.login()
run = finetuner.get_run('my_run')
```

## Saving your model
Show the user how to save their model when fine-tuning has completed.

Example:

"If your run has finished successfully, you can save fine-tuned models in the following way:"
```python
run.save_artifact('my_model')
```

## Evaluating your model
Explain to the user how they can track the performance of the model(s) they have fine-tuned in their runs. If this is not implemented yet, show the user an example log and how they might deduce model performance from this log.