(start-finetuner)=
# Run Job

Now you should have your training data and evaluation data (optional) prepared as CSV files or {class}`~docarray.array.document.DocumentArray`s,
and have selected your backbone model.

Up until now, you have worked locally to prepare a dataset and select our model. From here on out, you will send your processes to the cloud!

## Submit a Finetuning Job to the cloud

To start fine-tuning, you can call:

```python
import finetuner
from docarray import DocumentArray

train_data = 'path/to/some/data.csv'

run = finetuner.fit(
    model='efficientnet_b0',
    train_data=train_data
)
print(f'Run name: {run.name}')
print(f'Run status: {run.status()}')
```

You'll see something like this in the terminal, with a different run name:

```bash
Run name: vigilant-tereshkova
Run status: CREATED
```

During fine-tuning,
the run status changes from:
1. CREATED: the {class}`~finetuner.run.Run` has been created and submitted to the job queue.
2. STARTED: the job is in progress
3. FINISHED: the job finished successfully, model has been sent to Jina AI Cloud.
4. FAILED: the job failed, please check the logs for more details.

## Advanced configurations
Beyond the simplest use case,
Finetuner gives you the flexibility to set hyper-parameters explicitly:

```python
import finetuner
from docarray import DocumentArray
from finetuner.data import CSVOptions

train_data = 'path/to/some/train_data.csv'
eval_data = 'path/to/some/eval_data.csv'

# Create an experiment
finetuner.create_experiment(name='finetune-flickr-dataset')

run = finetuner.fit(
    model='efficientnet_b0',
    train_data=train_data,
    eval_data=eval_data, 
    run_name='finetune-flickr-dataset-efficientnet-1',
    description='this is a trial run on flickr8k dataset with efficientnet b0.',
    experiment_name='finetune-flickr-dataset', # Link to the experiment created above.
    model_options={}, # Additional options to pass to the model constructor.
    loss='TripletMarginLoss', # Use CLIPLoss for CLIP fine-tuning.
    miner='TripletMarginMiner',
    miner_options={'margin': 0.2}, # Additional options for the miner constructor.
    optimizer='Adam',
    optimizer_options={'weight_decay': 0.01}, # Additional options for the optimizer.
    learning_rate = 1e-4,
    epochs=5,
    batch_size=128,
    scheduler_step='batch',
    freeze=False, # If applied will freeze the embedding model, only train the MLP.
    output_dim=512, # Attach a MLP on top of embedding model.
    device='cuda',
    num_workers=4,
    to_onnx=False,  # If set, please pass `is_onnx` when making inference.
    csv_options=CSVOptions(),  # Additional options for reading data from a CSV file.
    public=False,  # If set, anyone has the artifact id can download your fine-tuned model.
    num_items_per_class=4,  # How many items per class to include in a batch.
)
```

### Loss functions

The loss function determines the training objective.
The type of loss function which is most suitable for your task depends heavily on the task your training for.
For many retrieval tasks, the `TripletMarginLoss` is a good choice.

```{Important}
Please check the [developer reference](../../api/finetuner/#finetuner.fit) to get the available options for `loss`, `miner`, `optimizer` and `scheduler_step`.
```

### Configuration of the optimizer
Fintuner allows one to choose any of the optimizers provided by PyTorch.
By default, the `Adam` optimizer is selected.
To select a different one, you can specify its name in the `optimizer` attribute of the fit function.
Possible values are: `Adadelta`, `Adagrad`, `Adam`, `AdamW`, `SparseAdam`, `Adamax`, `ASGD`, `LBFGS`, `NAdam`, `RAdam`, `RMSprop`, `Rprop`, and `SGD`.

Finetuner configures the learning rate of the optimizer by using the value of the `lr` option.
If you want to pass more parameters to the optimizer, you can specify them via `optimizer_options`.
For example, you can enable the weight decay of the Adam optimizer to penalize high weights in the model by setting `optimizer_options={'weight_decay':0.01}`.

For detailed documentation of the optimizers and their parameters, please take a look at the [PyTorch documentation](https://pytorch.org/docs/stable/optim.html).

```{admonition} Choosing the right learning rate and number of epochs
:class: hint
The learning rate determines how strong the weights are adjusted after processing a batch of training data.
In general, you should choose a low learning rate (`1e-6` to `1e-4`) for fine-tuning.
Otherwise, it could happen, that your model overfits on the training data and forgets the knowledge learned during pre-training.
Similarly, two or three epochs (number of passes thorough the training data) are often enough for a fine-tuning job. 
```


### Construction of Training Batches

The training of your model is done in batches.
The `batch_size` parameter determines the number of items per batch.
Finetuner constructs batches so that each batch contains the same number of classes and
as many items per class as configured via the `num_items_per_class` parameter.
However, if it is not possible, e.g., because `batch_size` is not dividable by
`num_items_per_class` or the training dataset does not contain enough classes,
Finetuner tries to choose a similar value for `num_items_per_class` which is working.