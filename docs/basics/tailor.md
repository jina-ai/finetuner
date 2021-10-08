# Tailor

Tailor is a component of Finetuner. It converts any {term}`general model` into an {term}`embedding model`. Given a general model (written from scratch, or from Pytorch/Keras/Huggingface model zoo), Tailor does micro-operations on the model architecture and outputs an embedding model for the {term}`Tuner`. 

Given a general model, Tailor does the following things:
- Finding all dense layers by iterating over layers;
- Removing all layers after a selected dense layer (including itself);
- (Optional) freezing the remaining layers;
- Add a new dense layer with the desired output dimensions as the last layer.


## Convert method

Tailor provides a high-level API `finetuner.tailor.convert()`, which can be used as following:

```python
from finetuner.tailor import to_embedding_model

to_embedding_model()
```

```{tip}
In general, you do not need to call `convert` manually. It is called by `finetuner.fit`. 
```


## Examples

### Simple MLP

### Simple Bi-LSTM

### VGG16