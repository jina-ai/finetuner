(hyperparameters)=
# {octicon}`tools` Hyperparameters

Beyond the simplest use case,
Finetuner gives you the flexibility to set hyper-parameters explicitly:

```python
import finetuner
from finetuner import DocumentArray
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
    scheduler='linear', # Use a linear scheduler to adjust the learning rate.
    scheduler_options={}, # Additional options for the scheduler.
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

Several of these parameters are explained in detail in their own sections,
the rest of this section will explain how best to use the ones that aren't.

## Learning Rate
The learning rate affects how much new training data affects the weights of the model you are training,
a higher learning rate results in a greater change.
It is important to choose an appropriate learning rate when training;
too high and you will overfit on your training data and you will lose the knowledge gained from pre-training,
too low and the model will not chage enough to produce significant improvement.  
Typically, an apropriate learning rate lies in the range of `1e-4` to `1e-6`. To see a typical learning rate for the different models and tasks we support, check out our [examples](#todo: figure out what to put here).

## Epochs
Each epoch is a single round of training, in each of these epochs, the model is trained on the entire training
data, so training for a higher number of epochs will result in a larger change in the model's performance.
Typically 2-3 epochs are enough for a fine-tuning job.

## Batch Size
During fine-tuning, your training data is split up into batches and is trained on one batch at a time.
The `batch_size` parameter can be used to configure the size of each batch.
A larger `batch_size` results in faster training, though too large a `batch_size` can result
in out of memory errors. Typically, a `batch_size` of 64 or 128 are good options when you
are unsure of how high you can set this value, however you can also choose to not set the `batch_size`
at all, in which case the highest possible value will be calculated for you automatically.

## Learning Rate Scheduler
You can configure Finetuner to use a learning rate scheduler.
The scheduler is used to adjust the learning rate during training.
If no scheduler is configured, the learning rate is constant during training.
When a scheduler is configured, the learning rate is adjusted after each batch by default.
Alternatively, one can set `scheduler_options = {'scheduler_step': 'epoch'}` to adjust the learning rate after
each epoch, when no scheduler is configured, this `scheduler_options` parameter is ignored.
A scheduler usually has a warm-up phase, where the learning rate is increasing.
After that most learning rateschedulers decrease the learning rate.
For example, the `linear` scheduler decreases the learning rate linearly from the initial learning rate:
![Learning Rate](https://user-images.githubusercontent.com/6599259/221238105-ee294b7e-544a-4de8-8c92-0c61275f29bb.png)  
The length of the warm-up phase is configured via the `num_warmup_steps` option inside `scheduler_optons`.
By default, it is set to zero.

```{Important}
Please check the [developer reference](../../api/finetuner/#finetuner.fit) to get the available options for `scheduler`.
```

## Loss Functions and Miners
The loss function determines the training objective.
The type of loss function which is most suitable for your task depends heavily on the task your training for.
The `miner` parameter is used to specify the method used to construct batches.
Loss functions can have different requirements for the contents of each batch,
so the available miners can depend on the loss function used.

See the [loss functions](./loss-functions.md) page to see the available loss functions.

## Optimizer
The `optimizer` is the method used to update the weights of the model after each training batch.
The `optimizer` parameter, in combination with its accompanying `optimizer_options`
parameter can be configured in a number of ways.
See the [optimizers](./optimizers.md) page to see how to configure the optimizer.