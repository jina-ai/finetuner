# Tailor

Tailor is a component of Finetuner. It converts any {term}`general model` into an {term}`embedding model`. Given a general model (either written from scratch, or from Pytorch/Keras/Huggingface model zoo), Tailor does micro-operations on the model architecture and outputs an embedding model for the {term}`Tuner`. 

Given a general model with weights, Tailor *preserves its weights* and does (some of) the  following steps:
- finding all dense layers by iterating over layers;
- removing all layers after a selected dense layer;
- freezing the weights on the remaining layers;
- adding a new dense layer with the desired output dimensions as the last layer.

In the end, Tailor outputs an embedding model that can be fine-tuned in Tuner.

## `to_embedding_model` method

Tailor provides a high-level API `finetuner.tailor.to_embedding_model()`, which can be used as following:

```python
from finetuner.tailor import to_embedding_model

to_embedding_model(
    model: AnyDNN,
    layer_name: Optional[str] = None,
    output_dim: Optional[int] = None,
    freeze: bool = False,
    input_size: Optional[Tuple[int, ...]] = None,
    input_dtype: str = 'float32'
) -> AnyDNN
```

Here, `model` is the general model with loaded weights; `layer_name` is the selected bottleneck layer; `freeze` defines if to set weights of remaining layers as nontrainable parameters.

`input_size` and `input_dtype` are input type specification required by Pytorch and Paddle models. They are not required for Kersas models.

In general, you do not need to call `to_embedding_model` manually. You can directly use it via `finetuner.fit(..., to_embedding_model=True)`

## `display` method

Tailor also provides a helper function `finetuner.tailor.display()` that gives a table summary of a Keras/Pytorch/Paddle model.

Let's see how to use them in action.

## Examples

### Simple MLP

1. Let's first build a simple 2-layer perceptron with 128 and 32-dim output as layers via Pytorch/Keras/Paddle. 
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
   ````{tab} Pytorch
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
3. Say we want to get an embedding model that outputs 100-dimensional embeddings. One can simply do
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
   ````{tab} Pytorch
   
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
   One can see that Tailor adds an additional linear layer with 100-dimensional output at the end.
   
### Simple Bi-LSTM

1. Let's first build a simple Bi-directional LSTM with Pytorch/Keras/Paddle.
     ````{tab} PyTorch
     ```python
     import torch
   
     class LastCell(torch.nn.Module):
       def forward(self, x):
         out, _ = x
         return out[:, -1, :]
   
     model = torch.nn.Sequential(
       torch.nn.Embedding(num_embeddings=5000, embedding_dim=64),
       torch.nn.LSTM(64, 64, bidirectional=True, batch_first=True),
       LastCell(),
       torch.nn.Linear(in_features=2 * 64, out_features=32))
     ```
     ````
     ````{tab} Keras
     ```python
     import tensorflow as tf
   
     model = tf.keras.Sequential([
            tf.keras.layers.Embedding(input_dim=5000, output_dim=64),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
            tf.keras.layers.Dense(32)])
     ```
     ````
     ````{tab} Paddle
     ```python
     import paddle
   
     class LastCell(paddle.nn.Layer):
        def forward(self, x):
            out, _ = x
            return out[:, -1, :]
   
     embed_model = paddle.nn.Sequential(
        paddle.nn.Embedding(num_embeddings=5000, embedding_dim=64),
        paddle.nn.LSTM(64, 64, direction='bidirectional'),
        LastCell(),
        paddle.nn.Linear(in_features=2 * 64, out_features=32))
     ```
     ````

2. Let's use `display` to look at the layer information.
   ````{tab} Pytorch
   ```python
   from finetuner.tailor import display
    
   display(model, input_size=(100, ), input_dtype='int64')
   ```
   
   ```console
     name          output_shape_display         nb_params   trainable  
    ────────────────────────────────────────────────────────────────── 
     embedding_1   [100, 64]                    320000      True       
     lstm_2        [[[2, 2, 64], [2, 2, 64]]]   66560       False      
     lastcell_3    [128]                        0           False      
     linear_4      [32]                         4128        True       
   ```
   ````
   ````{tab} Keras
    ```python
    from finetuner.tailor import display
    
    display(model)
    ```
   
   ```console
     name            output_shape_display   nb_params   trainable  
    ────────────────────────────────────────────────────────────── 
     embedding       [None, 64]             320000      True       
     bidirectional   [128]                  66048       True       
     dense           [32]                   4128        True       
   ```   
   
   ````
   ````{tab} Paddle
   ```python
   from finetuner.tailor import display
    
   display(model, input_size=(100, ), input_dtype='int64')
   ```
   
   ```console
     name          output_shape_display         nb_params   trainable  
    ────────────────────────────────────────────────────────────────── 
     embedding_1   [100, 64]                    320000      True       
     lstm_2        [[[2, 2, 64], [2, 2, 64]]]   66560       True       
     lastcell_3    [128]                        0           False      
     linear_4      [32]                         4128        True       
   ```      
   ````
3. Say we want to get an embedding model that outputs 100-dimensional embeddings. But this time, we want to directly concat this layer after LSTM, and freeze all previous layers. One can use `layer_name` and `freeze` to solve this problem:
   ````{tab} Pytorch
   
   ```python
   from finetuner.tailor import to_embedding_model
    
   embed_model = to_embedding_model(model,
                                 layer_name='lastcell_3',
                                 freeze=True,
                                 output_dim=100,
                                 input_size=(100,), input_dtype='int64')
   ```
   
   ```console
     name          output_shape_display         nb_params   trainable  
    ────────────────────────────────────────────────────────────────── 
     embedding_1   [100, 64]                    320000      False      
     lstm_2        [[[2, 2, 64], [2, 2, 64]]]   66560       False      
     lastcell_3    [128]                        0           False      
     linear_4      [100]                        12900       True       
   ```
   ````
   ````{tab} Keras
   ```python
   from finetuner.tailor import to_embedding_model

   embed_model = to_embedding_model(model,
                                    layer_name='bidirectional',
                                    freeze=True,
                                    output_dim=100)
   ```
   
   ```console
     name            output_shape_display   nb_params   trainable  
    ────────────────────────────────────────────────────────────── 
     embedding       [None, 64]             320000      False      
     bidirectional   [128]                  66048       False      
     dense_1         [100]                  12900       True       
   ```   
   
   ````
   ````{tab} Paddle
   ```python
   from finetuner.tailor import to_embedding_model
    
   embed_model = to_embedding_model(model,
                                 layer_name='lastcell_3',
                                 freeze=True,
                                 output_dim=100,
                                 input_size=(100,), input_dtype='int64')
   ```
   ```console
     name          output_shape_display         nb_params   trainable  
    ────────────────────────────────────────────────────────────────── 
     embedding_1   [100, 64]                    320000      False      
     lstm_2        [[[2, 2, 64], [2, 2, 64]]]   66560       False      
     lastcell_3    [128]                        0           False      
     linear_4      [100]                        12900       True       
   ```      
   ```` 


### VGG16