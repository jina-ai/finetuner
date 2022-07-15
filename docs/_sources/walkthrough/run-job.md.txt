(start-finetuner)=
# Run Job

Now you should have your training data and evaluation data (optional) prepared as {class}`~docarray.array.document.DocumentArray`s,
and have selected your backbone model.

Up until now, you have worked locally to prepare a dataset and select our model. From here on out, you will send your processes to the cloud!

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

You'll see something this in the terminal, with a different run name:

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
    optimizer='Adam',
    learning_rate = 1e-4,
    epochs=10,
    batch_size=128,
    scheduler_step='batch',
    freeze=False, # If applied will freeze the embedding model, only train the MLP.
    output_dim=512, # Attach a MLP on top of embedding model.
    cpu=False,
    num_workers=4,
)
```

```{Important}
Please check the [developer reference](../../api/finetuner/#finetuner.fit) to get the available options for `loss`, `miner`, `optimizer` and `scheduler_step`.
```