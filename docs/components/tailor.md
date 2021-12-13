# Tailor

Tailor is a component of Finetuner. It converts any {term}`general model` into an {term}`embedding model`. Given a general model (either written from scratch, or from PyTorch/Keras/Huggingface model zoo), Tailor performs micro-operations on the model architecture and outputs an embedding model for the {term}`Tuner`. 

Given a general model with weights, Tailor *preserves its weights* and performs (some of) the following steps:
- finding all dense layers by iterating over layers;
- chopping off all layers after a certain dense layer;
- freezing weights of specific layers.
- adding a new bottleneck layer with the desired output dimensions as the last layer.

```{figure} tailor-feature.svg
:align: center
```

Finally, Tailor outputs an embedding model that can be fine-tuned in Tuner.

## `to_embedding_model` method

Tailor provides a high-level API `finetuner.tailor.to_embedding_model()`, which can be used as follows:

```python
from finetuner.tailor import to_embedding_model

to_embedding_model(
    model: AnyDNN,
    layer_name: Optional[str] = None,
    input_size: Optional[Tuple[int, ...]] = None,
    input_dtype: str = 'float32',
    freeze: Union[bool, List[str]] = False,
) -> AnyDNN
```

Here, `model` is the general model with loaded weights;
`layer_name` is the selected embedding layer, all the layers after `layer_name` will be replaced with Identity layer.
`freeze` determines if specified layer is trainable. It accepts a `bool` value or a list of `str` as inputs.
if `freeze` is `True`, tailor will freeze all layers in the `model`. Otherwise, tailor will freeze weights by layer names.
You can visualize model structure and layer names with `display` method.
It will be introduced in the following section.

`input_size` and `input_dtype` are input type specification required by PyTorch and Paddle models. They are not required for Keras models.

In general, you do not need to call `to_embedding_model` manually. You can use it directly via `finetuner.fit(..., to_embedding_model=True)`

(display-method)=
## `display` method

Tailor also provides a helper function `finetuner.display()` that gives a table summary of a Keras/PyTorch/Paddle model.

Let's see how to use them in action.

## Examples

### Simple MLP

1. Let's first build a simple 2-layer perceptron with 128 and 32-dim output as layers via PyTorch/Keras/Paddle. 
    ````{tab} PyTorch
    ```python
    import torch
   
    model = torch.nn.Sequential(
          torch.nn.Flatten(),
          torch.nn.Linear(in_features=28 * 28, out_features=128),
          torch.nn.ReLU(),
          torch.nn.Linear(in_features=128, out_features=32))
    ```
   
    ````
    ````{tab} Keras
    ```python
    import tensorflow as tf
   
    model = tf.keras.Sequential([
                tf.keras.layers.Flatten(input_shape=(28, 28)),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(32)
            ])
    ```
    ````
    ````{tab} Paddle
    ```python
    import paddle
   
    model = paddle.nn.Sequential(
          paddle.nn.Flatten(),
          paddle.nn.Linear(in_features=28 * 28, out_features=128),
          paddle.nn.ReLU(),
          paddle.nn.Linear(in_features=128, out_features=32))
    ```
   
    ````
2. Let's use `display` to look at the layer information.
   ````{tab} PyTorch
   ```python
   from finetuner.tailor import display
    
   display(model, input_size=(28, 28))
   ```
   
   ```console
     name        output_shape_display   nb_params   trainable  
    ────────────────────────────────────────────────────────── 
     flatten_1   [784]                  0           False      
     linear_2    [128]                  100480      True       
     relu_3      [128]                  0           False      
     linear_4    [32]                   4128        True          
   ```
   ````
   ````{tab} Keras
    ```python
    from finetuner.tailor import display
    
    display(model)
    ```
   
   ```console
     name      output_shape_display   nb_params   trainable  
    ──────────────────────────────────────────────────────── 
     flatten   [784]                  0           False       
     dense     [128]                  100480      True       
     dense_1   [32]                   4128        True       
   ```   
   
   ````
   ````{tab} Paddle
   ```python
   from finetuner.tailor import display
    
   display(model, input_size=(28, 28))
   ```
   
   ```console
     name        output_shape_display   nb_params   trainable  
    ────────────────────────────────────────────────────────── 
     flatten_1   [784]                  0           False      
     linear_2    [128]                  100480      True       
     relu_3      [128]                  0           False      
     linear_4    [32]                   4128        True       
   ```      
   ````
3. Say we want to get an embedding model that outputs 100-dimensional embeddings. You can simply do:
   ```python
   from finetuner.tailor import to_embedding_model
    
   embed_model = to_embedding_model(model,
                      output_dim=100,
                      input_size=(28, 28))
   ```
4. Now let's look at the layer information again using `display`.
   ```python
   from finetuner.tailor import display
    
   display(model, input_size=(28, 28))
   ```
   ````{tab} PyTorch
   
   ```console
     name        output_shape_display   nb_params   trainable  
    ────────────────────────────────────────────────────────── 
     flatten_1   [784]                  0           False      
     linear_2    [128]                  100480      True       
     relu_3      [128]                  0           False      
     linear_4    [32]                   4128        True       
     linear_5    [100]                  3300        True       
   ```
   ````
   ````{tab} Keras
   ```console
     name            output_shape_display   nb_params   trainable  
    ────────────────────────────────────────────────────────────── 
     flatten_input   []                     0           False      
     flatten         [784]                  0           False      
     dense           [128]                  100480      True       
     dense_1         [32]                   4128        True       
     dense_2         [100]                  3300        True      
   ```   
   
   ````
   ````{tab} Paddle
   
   ```console
     name        output_shape_display   nb_params   trainable  
    ────────────────────────────────────────────────────────── 
     flatten_1   [784]                  0           False      
     linear_2    [128]                  100480      True       
     relu_3      [128]                  0           False      
     linear_4    [32]                   4128        True       
     linear_5    [100]                  3300        True       
   ```      
   ````
   You can see that Tailor adds an additional linear layer with 100-dimensional output at the end.
   


## Tips

- For PyTorch/Paddle models, having the correct `input_size` and `input_dtype` is fundamental to use `to_embedding_model` and `display`.
- You can chop off layers and concat new layers afterward. To get the accurate layer name, you can first use `display` to list all layers. 
- Different frameworks may give different layer names. Often, PyTorch and Paddle layer names are consistent.
