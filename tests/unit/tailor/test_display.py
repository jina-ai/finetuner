import pytest

from finetuner.tailor import display


def get_mlp(framework):
    if framework == 'pytorch':
        import torch

        return torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=28 * 28, out_features=128),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=128, out_features=32),
        )
    elif framework == 'keras':
        import tensorflow as tf

        return tf.keras.Sequential(
            [
                tf.keras.layers.Flatten(input_shape=(28, 28)),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(32),
            ]
        )
    elif framework == 'paddle':
        import paddle

        return paddle.nn.Sequential(
            paddle.nn.Flatten(),
            paddle.nn.Linear(in_features=28 * 28, out_features=128),
            paddle.nn.ReLU(),
            paddle.nn.Linear(in_features=128, out_features=32),
        )


def get_mlp_display(framework):
    if framework == 'pytorch':
        return '''
  name        output_shape_display   nb_params   trainable  
 ────────────────────────────────────────────────────────── 
  flatten_1   [784]                  0           False      
  linear_2    [128]                  100480      True       
  relu_3      [128]                  0           False      
  linear_4    [32]                   4128        True       
    '''
    elif framework == 'paddle':
        return '''
  name        output_shape_display   nb_params   trainable  
 ────────────────────────────────────────────────────────── 
  flatten_1   [784]                  0           False      
  linear_2    [128]                  100480      True       
  relu_3      [128]                  0           False      
  linear_4    [32]                   4128        True       
       '''
    elif framework == 'keras':
        return '''
  name      output_shape_display   nb_params   trainable  
 ──────────────────────────────────────────────────────── 
  flatten   [784]                  0           False      
  dense     [128]                  100480      True       
  dense_1   [32]                   4128        True       
           '''


def get_lstm(framework):
    if framework == 'pytorch':
        import torch

        class LastCell(torch.nn.Module):
            def forward(self, x):
                out, _ = x
                return out[:, -1, :]

        model = torch.nn.Sequential(
            torch.nn.Embedding(num_embeddings=5000, embedding_dim=64),
            torch.nn.LSTM(64, 64, bidirectional=True, batch_first=True),
            LastCell(),
            torch.nn.Linear(in_features=2 * 64, out_features=32),
        )
        return model
    elif framework == 'keras':
        import tensorflow as tf

        model = tf.keras.Sequential(
            [
                tf.keras.layers.Embedding(input_dim=5000, output_dim=64),
                tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
                tf.keras.layers.Dense(32),
            ]
        )
        return model
    elif framework == 'paddle':
        import paddle

        class LastCell(paddle.nn.Layer):
            def forward(self, x):
                out, _ = x
                return out[:, -1, :]

        model = paddle.nn.Sequential(
            paddle.nn.Embedding(num_embeddings=5000, embedding_dim=64),
            paddle.nn.LSTM(64, 64, direction='bidirectional'),
            LastCell(),
            paddle.nn.Linear(in_features=2 * 64, out_features=32),
        )
        return model


def get_lstm_display(framework):
    if framework == 'pytorch':
        return '''
  name          output_shape_display         nb_params   trainable  
 ────────────────────────────────────────────────────────────────── 
  embedding_1   [100, 64]                    320000      True       
  lstm_2        [[[2, 2, 64], [2, 2, 64]]]   66560       False      
  lastcell_3    [128]                        0           False      
  linear_4      [32]                         4128        True       
    '''
    elif framework == 'paddle':
        return '''
  name          output_shape_display         nb_params   trainable  
 ────────────────────────────────────────────────────────────────── 
  embedding_1   [100, 64]                    320000      True       
  lstm_2        [[[2, 2, 64], [2, 2, 64]]]   66560       True       
  lastcell_3    [128]                        0           False      
  linear_4      [32]                         4128        True       
       '''
    elif framework == 'keras':
        return '''
  name            output_shape_display   nb_params   trainable  
 ────────────────────────────────────────────────────────────── 
  embedding       [None, 64]             320000      True       
  bidirectional   [128]                  66048       True       
  dense           [32]                   4128        True       
           '''


def get_vgg(framework):
    if framework == 'pytorch':
        import torchvision.models as models

        model = models.vgg16()
    elif framework == 'keras':
        import tensorflow as tf

        model = tf.keras.applications.vgg16.VGG16()
    elif framework == 'paddle':
        import paddle

        model = paddle.vision.models.vgg16(pretrained=False)
    else:
        raise NotImplementedError
    return model


def get_vgg_display(framework):
    if framework == 'pytorch':
        return '''
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
       '''
    elif framework == 'paddle':
        return '''
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
          '''
    elif framework == 'keras':
        return '''
  name           output_shape_display   nb_params   trainable  
 ───────────────────────────────────────────────────────────── 
  input_1        []                     0           False      
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
              '''


@pytest.mark.parametrize('framework', ('paddle', 'pytorch', 'keras'))
def test_display_mlp_original(framework, capsys):
    model = get_mlp(framework)
    _display = get_mlp_display(framework)
    display(model, input_size=(28, 28))
    assert _display.strip() in capsys.readouterr().out.strip()


@pytest.mark.parametrize('framework', ('paddle', 'pytorch', 'keras'))
def test_display_lstm_original(framework, capsys):
    model = get_lstm(framework)
    _display = get_lstm_display(framework)
    display(model, input_size=(100,), input_dtype='int64')
    assert _display.strip() in capsys.readouterr().out.strip()


@pytest.mark.parametrize('framework', ('paddle', 'pytorch', 'keras'))
def test_display_vgg_original(framework, capsys):
    model = get_vgg(framework)
    _display = get_vgg_display(framework)
    display(model, input_size=(3, 224, 224))
    # assert _display.strip() in capsys.readouterr().out.strip()
