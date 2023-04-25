(hyperparameters)=
# {octicon}`tools` Hyperparameters

Several of the `finetuner.fit` function's parameters are _hyper-parameters_: values that do not change the methods used during training but control the overall behavior and settings of the training process.
These fall into four areas:
- [Setting the learning rate and using an optimizer.](./optimizers.md)
- [Setting a loss function and using a miner.](./loss-functions.md)
- Setting the batch size.
- Choosing how many epochs to train.

For the first two, see the pages linked above. For epochs and batch sizes, read the discussion below.

## Epochs
An epoch is a single round of training in which the model is presented with each item of training data,
and after each item, weights are updated.
Training for more epochs will result in a larger change in the model's performance but will take more time.
Training on too many epochs can result in overfitting, so it is important to choose a reasonable number.
For transformer models, training can be completed with only 1-3 epochs while still producing a good improvement, whereas other models might profit from training for more epochs, typically between 4 and 10.

## Batch Size
During fine-tuning, training data is split into batches, and training takes place one batch at a time.
The `batch_size` parameter sets the size of each batch.
A larger `batch_size` results in faster training, though too large a `batch_size` can result
in out-of-memory errors.
The optimal `batch_size` is dependent on the model you are using and the contents of your training data.
A `batch_size` of 64 or 128 is generally reasonably safe if you don't know how high you can set this value. However, if you do not set the `batch_size` at all,
Finetuner will determine the highest possible value for your system and set it automatically.

```{Important}
CLIP models are usually larger than other models, with the `vit-large-en` model only being able to be trained safely with a `batch_size` of 8. If you are not sure what `batch_size` to use for a model, we recommend not setting it and allowing the `batch_size` to be calculated automatically.
```

