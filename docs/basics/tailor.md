# Tailor

Tailor is a component of Finetuner. It converts any {term}`general model` into an {term}`embedding model`. Given a general model (either written from scratch, or from Pytorch/Keras/Huggingface model zoo), Tailor does micro-operations on the model architecture and outputs an embedding model for the {term}`Tuner`. 

Given a general model with weights, Tailor *preserves its weights* and does (some of) the  following steps:
- finding all dense layers by iterating over layers;
- chopping off all layers after a certain dense layer;
- freezing the weights on the remaining layers;
- adding a new dense layer with the desired output dimensions as the last layer.

```{figure} tailor-feature.svg
:align: center
```

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
   
     model = paddle.nn.Sequential(
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
3. Say we want to get an embedding model that outputs 100-dimensional embeddings. But this time, we want to directly concat this layer after LSTM, and freeze all previous layers. One can use `layer_name` and `freeze` to solve this problem. In Pytorch and Paddle implementation, the layer name is `lastcell_3`; in Keras the layer name is `bidirectional`.
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
   One can observe the last linear layer is replaced from a 32-dimensional output to a 100-dimensional output. Also, the weights of all layers except the last layers are frozen and not trainable. 


### Pretrained VGG16 model

Apart from building model on your own and then tailor it, Tailor can work directly on pretrained models. In this example, we load a pretrained VGG16 model and tailor it into an embedding model. 


1. Let's first load a pretrained VGG16 from Pytorch/Keras/Paddle model zoo.
     ````{tab} PyTorch
     ```python
     import torchvision.models as models

     model = models.vgg16()
     ```
     ````
     ````{tab} Keras
     ```python
     import tensorflow as tf
   
     model = tf.keras.applications.vgg16.VGG16()
     ```
     ````
     ````{tab} Paddle
     ```python
     import paddle

     model = paddle.vision.models.vgg16()
     ```
     ````

2. Let's use `display` to look at the layer information.
   ````{tab} Pytorch
   ```python
   from finetuner.tailor import display
    
   display(model, input_size=(3, 224, 224))
   ```
   
   ```console
     name                   output_shape_display   nb_params   trainable  
    ───────────────────────────────────────────────────────────────────── 
     conv2d_1               [64, 224, 224]         1792        True       
     relu_2                 [64, 224, 224]         0           False      
     conv2d_3               [64, 224, 224]         36928       True       
     relu_4                 [64, 224, 224]         0           False      
     maxpool2d_5            [64, 112, 112]         0           False      
     conv2d_6               [128, 112, 112]        73856       True       
     relu_7                 [128, 112, 112]        0           False      
     conv2d_8               [128, 112, 112]        147584      True       
     relu_9                 [128, 112, 112]        0           False      
     maxpool2d_10           [128, 56, 56]          0           False      
     conv2d_11              [256, 56, 56]          295168      True       
     relu_12                [256, 56, 56]          0           False      
     conv2d_13              [256, 56, 56]          590080      True       
     relu_14                [256, 56, 56]          0           False      
     conv2d_15              [256, 56, 56]          590080      True       
     relu_16                [256, 56, 56]          0           False      
     maxpool2d_17           [256, 28, 28]          0           False      
     conv2d_18              [512, 28, 28]          1180160     True       
     relu_19                [512, 28, 28]          0           False      
     conv2d_20              [512, 28, 28]          2359808     True       
     relu_21                [512, 28, 28]          0           False      
     conv2d_22              [512, 28, 28]          2359808     True       
     relu_23                [512, 28, 28]          0           False      
     maxpool2d_24           [512, 14, 14]          0           False      
     conv2d_25              [512, 14, 14]          2359808     True       
     relu_26                [512, 14, 14]          0           False      
     conv2d_27              [512, 14, 14]          2359808     True       
     relu_28                [512, 14, 14]          0           False      
     conv2d_29              [512, 14, 14]          2359808     True       
     relu_30                [512, 14, 14]          0           False      
     maxpool2d_31           [512, 7, 7]            0           False      
     adaptiveavgpool2d_32   [512, 7, 7]            0           False      
     linear_33              [4096]                 102764544   True       
     relu_34                [4096]                 0           False      
     dropout_35             [4096]                 0           False      
     linear_36              [4096]                 16781312    True       
     relu_37                [4096]                 0           False      
     dropout_38             [4096]                 0           False      
     linear_39              [1000]                 4097000     True   
   ```
   ````
   ````{tab} Keras
    ```python
    from finetuner.tailor import display
    
    display(model)
    ```
   
   ```console
     name           output_shape_display   nb_params   trainable  
    ───────────────────────────────────────────────────────────── 
     block1_conv1   [224, 224, 64]         1792        True       
     block1_conv2   [224, 224, 64]         36928       True       
     block1_pool    [112, 112, 64]         0           False      
     block2_conv1   [112, 112, 128]        73856       True       
     block2_conv2   [112, 112, 128]        147584      True       
     block2_pool    [56, 56, 128]          0           False      
     block3_conv1   [56, 56, 256]          295168      True       
     block3_conv2   [56, 56, 256]          590080      True       
     block3_conv3   [56, 56, 256]          590080      True       
     block3_pool    [28, 28, 256]          0           False      
     block4_conv1   [28, 28, 512]          1180160     True       
     block4_conv2   [28, 28, 512]          2359808     True       
     block4_conv3   [28, 28, 512]          2359808     True       
     block4_pool    [14, 14, 512]          0           False      
     block5_conv1   [14, 14, 512]          2359808     True       
     block5_conv2   [14, 14, 512]          2359808     True       
     block5_conv3   [14, 14, 512]          2359808     True       
     block5_pool    [7, 7, 512]            0           False      
     flatten        [25088]                0           False      
     fc1            [4096]                 102764544   True       
     fc2            [4096]                 16781312    True       
     predictions    [1000]                 4097000     True       
   ```   
   
   ````
   ````{tab} Paddle
   ```python
   from finetuner.tailor import display
    
   display(model, input_size=(3, 224, 224))
   ```
   
   ```console
     name           output_shape_display   nb_params   trainable  
    ───────────────────────────────────────────────────────────── 
     conv2d_1       [64, 224, 224]         1792        True       
     conv2d_3       [64, 224, 224]         36928       True       
     maxpool2d_5    [64, 112, 112]         0           False      
     conv2d_6       [128, 112, 112]        73856       True       
     conv2d_8       [128, 112, 112]        147584      True       
     maxpool2d_10   [128, 56, 56]          0           False      
     conv2d_11      [256, 56, 56]          295168      True       
     conv2d_13      [256, 56, 56]          590080      True       
     conv2d_15      [256, 56, 56]          590080      True       
     maxpool2d_17   [256, 28, 28]          0           False      
     conv2d_18      [512, 28, 28]          1180160     True       
     conv2d_20      [512, 28, 28]          2359808     True       
     conv2d_22      [512, 28, 28]          2359808     True       
     maxpool2d_24   [512, 14, 14]          0           False      
     conv2d_25      [512, 14, 14]          2359808     True       
     conv2d_27      [512, 14, 14]          2359808     True       
     conv2d_29      [512, 14, 14]          2359808     True       
     maxpool2d_31   [512, 7, 7]            0           False      
     linear_33      [4096]                 102764544   True       
     linear_36      [4096]                 16781312    True       
     linear_39      [1000]                 4097000     True  
   ```      
   ````
3. Say we want to get an embedding model that outputs 100-dimensional embeddings. This time we want to remove all existing dense layers, and freeze all previous layers, then concat a new 100-dimensional dense output to the mode. To achieve that, 
   ````{tab} Pytorch
   
   ```python
   from finetuner.tailor import to_embedding_model
    
   embed_model = to_embedding_model(model,
                                 layer_name='linear_33',
                                 freeze=True,
                                 output_dim=100,
                                 input_size=(3, 224, 224))
   ```
   
   ```console
     name           output_shape_display   nb_params   trainable  
    ───────────────────────────────────────────────────────────── 
     conv2d_1       [64, 224, 224]         1792        False      
     conv2d_3       [64, 224, 224]         36928       False      
     maxpool2d_5    [64, 112, 112]         0           False      
     conv2d_6       [128, 112, 112]        73856       False      
     conv2d_8       [128, 112, 112]        147584      False      
     maxpool2d_10   [128, 56, 56]          0           False      
     conv2d_11      [256, 56, 56]          295168      False      
     conv2d_13      [256, 56, 56]          590080      False      
     conv2d_15      [256, 56, 56]          590080      False      
     maxpool2d_17   [256, 28, 28]          0           False      
     conv2d_18      [512, 28, 28]          1180160     False      
     conv2d_20      [512, 28, 28]          2359808     False      
     conv2d_22      [512, 28, 28]          2359808     False      
     maxpool2d_24   [512, 14, 14]          0           False      
     conv2d_25      [512, 14, 14]          2359808     False      
     conv2d_27      [512, 14, 14]          2359808     False      
     conv2d_29      [512, 14, 14]          2359808     False      
     maxpool2d_31   [512, 7, 7]            0           False      
     linear_33      [4096]                 102764544   False      
     linear_34      [100]                  409700      True    
   ```
   ````
   ````{tab} Keras
   ```python
   from finetuner.tailor import to_embedding_model

   embed_model = to_embedding_model(model,
                                 layer_name='flatten',
                                 freeze=True,
                                 output_dim=100)
   ```
   
   ```console
     name           output_shape_display   nb_params   trainable  
    ───────────────────────────────────────────────────────────── 
     block1_conv1   [224, 224, 64]         1792        False      
     block1_conv2   [224, 224, 64]         36928       False      
     block1_pool    [112, 112, 64]         0           False      
     block2_conv1   [112, 112, 128]        73856       False      
     block2_conv2   [112, 112, 128]        147584      False      
     block2_pool    [56, 56, 128]          0           False      
     block3_conv1   [56, 56, 256]          295168      False      
     block3_conv2   [56, 56, 256]          590080      False      
     block3_conv3   [56, 56, 256]          590080      False      
     block3_pool    [28, 28, 256]          0           False      
     block4_conv1   [28, 28, 512]          1180160     False      
     block4_conv2   [28, 28, 512]          2359808     False      
     block4_conv3   [28, 28, 512]          2359808     False      
     block4_pool    [14, 14, 512]          0           False      
     block5_conv1   [14, 14, 512]          2359808     False      
     block5_conv2   [14, 14, 512]          2359808     False      
     block5_conv3   [14, 14, 512]          2359808     False      
     block5_pool    [7, 7, 512]            0           False      
     flatten        [25088]                0           False      
     dense          [100]                  2508900     True       
   ```   
   
   ````
   ````{tab} Paddle
   ```python
   from finetuner.tailor import to_embedding_model
    
   embed_model = to_embedding_model(model,
                                    layer_name='linear_33',
                                    freeze=True,
                                    output_dim=100,
                                    input_size=(3, 224, 224))
   ```
   ```console
     name           output_shape_display   nb_params   trainable  
    ───────────────────────────────────────────────────────────── 
     conv2d_1       [64, 224, 224]         1792        False      
     conv2d_3       [64, 224, 224]         36928       False      
     maxpool2d_5    [64, 112, 112]         0           False      
     conv2d_6       [128, 112, 112]        73856       False      
     conv2d_8       [128, 112, 112]        147584      False      
     maxpool2d_10   [128, 56, 56]          0           False      
     conv2d_11      [256, 56, 56]          295168      False      
     conv2d_13      [256, 56, 56]          590080      False      
     conv2d_15      [256, 56, 56]          590080      False      
     maxpool2d_17   [256, 28, 28]          0           False      
     conv2d_18      [512, 28, 28]          1180160     False      
     conv2d_20      [512, 28, 28]          2359808     False      
     conv2d_22      [512, 28, 28]          2359808     False      
     maxpool2d_24   [512, 14, 14]          0           False      
     conv2d_25      [512, 14, 14]          2359808     False      
     conv2d_27      [512, 14, 14]          2359808     False      
     conv2d_29      [512, 14, 14]          2359808     False      
     maxpool2d_31   [512, 7, 7]            0           False      
     linear_33      [4096]                 102764544   False      
     linear_34      [100]                  409700      True                                                              
   ```      
   ```` 
   One can observe the original last two linear layers are removed, and a new linear layer with 100-dimensional output is added at the end. Also, the weights of all layers except the last layers are frozen and not trainable. 

## Tips

- For Pytorch/Paddle models, having the correct `input_size` and `input_dtype` is fundamental to use `to_embedding_model` and `display`.
- One can chop-off layers and concat new layer afterward. To get the accurate layer name, one can first use `display` to list all layers. 
- Different frameworks may give different layer names. Often, Pytorch and Paddle layer names are consistent.