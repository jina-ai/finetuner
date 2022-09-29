(start-finetuner)=
# Run Job

Now you should have your training data and evaluation data (optional) prepared as {class}`~docarray.array.document.DocumentArray`s,
and have selected your backbone model.

Up until now, you have worked locally to prepare a dataset and select our model. From here on out, you will send your processes to the cloud!

## Submit a Finetuning Job to the cloud

To start fine-tuning, you can call:

```python
import finetuner
from docarray import DocumentArray

train_data = DocumentArray(...)

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
3. FINISHED: the job finished successfully, model has been sent to cloud storage.
4. FAILED: the job failed, please check the logs for more details.

## Advanced configurations
Beyond the simplest use case,
Finetuner gives you the flexibility to set hyper-parameters explicitly:

```python
import finetuner
from docarray import DocumentArray

train_data = DocumentArray(...)
eval_data = DocumentArray(...)

# Create an experiment
finetuner.create_experiment(name='finetune-flickr-dataset')

run = finetuner.fit(
    model='efficientnet_b0',
    train_data=train_data,
    eval_data=eval_data, 
    run_name='finetune-flickr-dataset-efficientnet-1',
    description='this is a trial run on flickr8k dataset with efficientnet b0.',
    experiment_name='finetune-flickr-dataset', # link to the experiment created above.
    model_options={}, # additional options to pass to the model constructor
    loss='TripletMarginLoss', # Use CLIPLoss for CLIP fine-tuning.
    miner='TripletMarginMiner',
    miner_options={'margin': 0.2}, # additional options for the miner constructor
    optimizer='Adam',
    optimizer_options={'weight_decay': 0.01}, # additional options for the optimizer
    learning_rate = 1e-4,
    epochs=5,
    batch_size=128,
    scheduler_step='batch',
    freeze=False, # If applied will freeze the embedding model, only train the MLP.
    output_dim=512, # Attach a MLP on top of embedding model.
    cpu=False,
    num_workers=4,
    to_onnx=False,  # If set, please pass `is_onnx` when making inference.
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

### Configuration of the miner

To filter the instances in a batch that are used to calculate the loss, you can use miners.
Finetuner allows you to use miners provided by the [PyTorch Metric Learning](https://kevinmusgrave.github.io/pytorch-metric-learning) framework.
To select a specific miner, you can pass its name to the fit function, e.g., `AngularMiner`, `TripletMarginMiner`, ...

Please note that the miner has to be compatible with the loss function you selected.
For instance, if you choose to train a model with the `TripleMarginLoss`, you can use the `TripletMarginMiner`.
While without this miner, all possible triples with an anchor, a positive, and a negative candidate are constructed, the miner reduces this set of triples.
Usually, only triples with hard negatives are selected where the distance between the positive and the negative example is inside a margin of `0.2`.
If you want to pass additional parameters to configure the miner, you can specify the `miner_options` parameter of the fit function.
For example, you can modify the size of the margin to `0.3` by setting `miner_options={'margin': 0.3}'`.
A detailed description of the miners and their parameters is specified in the [PyTorch Metric Learning documentation](https://kevinmusgrave.github.io/pytorch-metric-learning/miners/).

