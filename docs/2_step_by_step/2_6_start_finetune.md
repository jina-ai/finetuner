# Start fine-tuning

Now you should have your training data and evaluation data (optional) prepared as `DocumentArray`s,
and have selected your backbone model.

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

You'll see this in the terminal:

```bash
Run name: vigilant-tereshkova
Run status: CREATED
```

During fine-tuning,
the run status changes from:
1. CREATED: the `Run` has been created and submitted to the job queue.
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
    loss='TripletMarginLoss', # Use CLIPLoss for CLIP fine-tuning.
    miner='TripletMarginMiner',
    optimizer='Adam',
    learning_rate = 1e-4,
    epochs=10,
    batch_size=128,
    scheduler_step='batch',
    freeze=False, # If applied will freeze the embedding model, only train the MLP.
    output_dim=512, # Attach a MLP on top of embedding model.
    multi_modal=False, # CLIP specific.
    image_modality=None, # CLIP specific.
    text_modality=None, # CLIP specific.
    cpu=False,
    num_workers=4,
)
```

```{Important}
Please check the developer reference to get the available options for `loss`, `miner`, `optimizer` and `scheduler_step`.
```

```{Important}
CLIP specific parameters

`multi_modal`: Need to be set to True when you are fine-tuning CLIP since we are fine-tuning two models.
`image_modality` and `text_modality`: Need to be set to the corresponded value of the `modality` when you are creating training data.

For example:
```python
doc = Document(
    chunks=[
        Document(
            content='this is the text chunk',
            modality='text',
            tags={'finetuner_label': 1}
        ),
        Document(
            content='https://...picture.png',
            modality='image',
            tags={'finetuner_label': 1}
        ),
    ]
)
# in this case, image_modality and text_modality should be set correspondingly
finetuner.fit(
    ...,
    loss='CLIPLoss',
    image_modality='image',
    text_modality='text',
    multi_modal=True,
    ...,
)
```