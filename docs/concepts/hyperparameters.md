(hyperparameters)=
# {octicon}`tools` Hyperparameters

If needed,
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
    loss='TripletMarginLoss',
    miner='TripletMarginMiner',
    miner_options={'margin': 0.2}, # Additional options for the miner constructor.
    scheduler='linear', # Use a linear scheduler to adjust the learning rate.
    scheduler_options={}, # Additional options for the scheduler.
    optimizer='Adam',
    optimizer_options={'weight_decay': 0.01}, # Additional options for the optimizer.
    learning_rate=1e-4,
    epochs=5,
    batch_size=128,
    scheduler_step='batch',
    freeze=False, # If applied will freeze the embedding model, only train the MLP.
    output_dim=512, # Attach an MLP on top of embedding model.
    device='cuda',
    num_workers=4,
    to_onnx=False,  # If set, please pass `is_onnx` when performing inference.
    csv_options=CSVOptions(),  # Additional options for reading data from a CSV file.
    public=False,  # If set, anyone who has the artifact id can download your fine-tuned model.
    num_items_per_class=4,  # How many items per class to include in a batch.
)
```

Several of these parameters are explained in detail in their own sections,
the rest of this section will explain how to use the ones that aren't.

## Learning Rate
The learning rate affects how much each item of new training data can affect the weights of the model you are training each time the model sees it.
A higher learning rate means faster changes to the model, with greater change to the weights for each training epoch. A lower rate means changes will be slower.
It is essential to choose an appropriate learning rate when training:
too high, and you risk unstable model weights that never converge, overfitting your training data, and you may lose the knowledge gained from pre-training,
too low, and the model will take too long to learn and may not change enough to produce significant improvement.  
An appropriate learning rate is typically somewhere between `1e-4` and `1e-6`. To see typical learning rates for various models and tasks, check out our [examples](#todo: figure out what to put here).

## Epochs
An epoch is a single round of training in which the model is presented with each item of training data, and after each item, weights are updated. Training for more epochs will result in a larger change in the model's performance but will take more time.
Typically 3-10 epochs are enough for a fine-tuning job.

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
Alternatively, you can set `scheduler_options = {'scheduler_step': 'epoch'}` to adjust the learning rate after
each epoch. When no scheduler is configured, this `scheduler_options` parameter is ignored.
Schedulers usually have a warm-up phase, during which the learning rate increases.
After that, most learning rate schedulers decrease the learning rate.
For example, the `linear` scheduler decreases the learning rate linearly from the initial learning rate:
![Learning Rate](https://user-images.githubusercontent.com/6599259/221238105-ee294b7e-544a-4de8-8c92-0c61275f29bb.png)  
The length of the warm-up phase is configured via the `num_warmup_steps` option inside `scheduler_options`.
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
The `optimizer` determines the method used to update the weights of the model after each training batch.
The `optimizer` parameter, in combination with its accompanying `optimizer_options`
parameter can be configured in a number of ways.
See the [optimizers](./optimizers.md) page to see how to configure the optimizer.