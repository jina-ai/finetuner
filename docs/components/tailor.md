# Tailor

Tailor is a component of Finetuner. It converts any {term}`general model` into an {term}`embedding model`. Given a general model (either written from scratch, or from PyTorch/Keras/Huggingface model zoo), Tailor performs micro-operations on the model architecture and outputs an embedding model for the {term}`Tuner`. 

Given a general model with weights, Tailor *preserves its weights* and performs (some of) the following steps:
- finding all dense layers by iterating over layers;
- chopping off all layers after a certain dense layer;
- freezing weights of specific layers.
- adding a new bottleneck module on top of the embedding model.

```{figure} tailor-feature.svg
:align: center
```

Finally, Tailor outputs an embedding model that can be fine-tuned in Tuner.
To make the best use of Tailor, you could follow the journey of:
1. `display` model summary.
2. Select an embedding layer as your model output.
3. Decide your `freeze` strategy.
4. (optional) Attach a bottleneck module.

## `display` model summary

Tailor provides a helper function `finetuner.display()` that gives a table summary of a Keras/PyTorch/Paddle model.
Let's see how to create an embedding model with ResNet-50.

1. Load a pre-trained ResNet-50 via your favourite deep learning backend and call ``display``.

    ````{tab} PyTorch
    ```python
    import torchvision
    import finetuner as ft
   
    model = torchvision.models.resnet50(pretrained=True)
    ft.display(model, input_size=(3, 224, 224))
    ```
   
    ````
    ````{tab} Keras
    ```python
    import tensorflow as tf
    import finetuner as ft
   
    model = tf.keras.applications.ResNet50(weights='imagenet')
    ft.display(model)
    ```
    ````
    ````{tab} Paddle
    ```python
    import paddle
    import finetuner as ft
   
    model = paddle.vision.models.resnet50(pretrained=True)
    ft.display(model, input_size=(3, 224, 224))
    ```
    ````
2. Now you could see the model tabular summary in your console/notebook. The summary shows the overall layers, output shapes parameters and trainable layers.
   ````{tab} PyTorch
   
   ```console
     name                    output_shape_display   nb_params   trainable  
    ────────────────────────────────────────────────────────────────────── 
     conv2d_1                [64, 112, 112]         9408        True       
     batchnorm2d_2           [64, 112, 112]         128         True       
     relu_3                  [64, 112, 112]         0           False      
     maxpool2d_4             [64, 56, 56]           0           False      
     conv2d_5                [64, 56, 56]           4096        True       
     batchnorm2d_6           [64, 56, 56]           128         True       
     relu_7                  [64, 56, 56]           0           False      
     conv2d_8                [64, 56, 56]           36864       True       
     batchnorm2d_9           [64, 56, 56]           128         True       
     relu_10                 [64, 56, 56]           0           False      
     conv2d_11               [256, 56, 56]          16384       True       
     batchnorm2d_12          [256, 56, 56]          512         True       
     conv2d_13               [256, 56, 56]          16384       True       
     batchnorm2d_14          [256, 56, 56]          512         True       
     relu_15                 [256, 56, 56]          0           False      
     bottleneck_16           [256, 56, 56]          0           False      
     conv2d_17               [64, 56, 56]           16384       True       
     batchnorm2d_18          [64, 56, 56]           128         True       
     relu_19                 [64, 56, 56]           0           False      
     conv2d_20               [64, 56, 56]           36864       True       
     batchnorm2d_21          [64, 56, 56]           128         True       
     relu_22                 [64, 56, 56]           0           False      
     conv2d_23               [256, 56, 56]          16384       True       
     batchnorm2d_24          [256, 56, 56]          512         True       
     relu_25                 [256, 56, 56]          0           False      
     bottleneck_26           [256, 56, 56]          0           False      
     conv2d_27               [64, 56, 56]           16384       True       
     batchnorm2d_28          [64, 56, 56]           128         True       
     relu_29                 [64, 56, 56]           0           False      
     conv2d_30               [64, 56, 56]           36864       True       
     batchnorm2d_31          [64, 56, 56]           128         True       
     relu_32                 [64, 56, 56]           0           False      
     conv2d_33               [256, 56, 56]          16384       True       
     batchnorm2d_34          [256, 56, 56]          512         True       
     relu_35                 [256, 56, 56]          0           False      
     bottleneck_36           [256, 56, 56]          0           False      
     conv2d_37               [128, 56, 56]          32768       True       
     batchnorm2d_38          [128, 56, 56]          256         True       
     relu_39                 [128, 56, 56]          0           False      
     conv2d_40               [128, 28, 28]          147456      True       
     batchnorm2d_41          [128, 28, 28]          256         True       
     relu_42                 [128, 28, 28]          0           False      
     conv2d_43               [512, 28, 28]          65536       True       
     batchnorm2d_44          [512, 28, 28]          1024        True       
     conv2d_45               [512, 28, 28]          131072      True       
     batchnorm2d_46          [512, 28, 28]          1024        True       
     relu_47                 [512, 28, 28]          0           False      
     bottleneck_48           [512, 28, 28]          0           False      
     conv2d_49               [128, 28, 28]          65536       True       
     batchnorm2d_50          [128, 28, 28]          256         True       
     relu_51                 [128, 28, 28]          0           False      
     conv2d_52               [128, 28, 28]          147456      True       
     batchnorm2d_53          [128, 28, 28]          256         True       
     relu_54                 [128, 28, 28]          0           False      
     conv2d_55               [512, 28, 28]          65536       True       
     batchnorm2d_56          [512, 28, 28]          1024        True       
     relu_57                 [512, 28, 28]          0           False      
     bottleneck_58           [512, 28, 28]          0           False      
     conv2d_59               [128, 28, 28]          65536       True       
     batchnorm2d_60          [128, 28, 28]          256         True       
     relu_61                 [128, 28, 28]          0           False      
     conv2d_62               [128, 28, 28]          147456      True       
     batchnorm2d_63          [128, 28, 28]          256         True       
     relu_64                 [128, 28, 28]          0           False      
     conv2d_65               [512, 28, 28]          65536       True       
     batchnorm2d_66          [512, 28, 28]          1024        True       
     relu_67                 [512, 28, 28]          0           False      
     bottleneck_68           [512, 28, 28]          0           False      
     conv2d_69               [128, 28, 28]          65536       True       
     batchnorm2d_70          [128, 28, 28]          256         True       
     relu_71                 [128, 28, 28]          0           False      
     conv2d_72               [128, 28, 28]          147456      True       
     batchnorm2d_73          [128, 28, 28]          256         True       
     relu_74                 [128, 28, 28]          0           False      
     conv2d_75               [512, 28, 28]          65536       True       
     batchnorm2d_76          [512, 28, 28]          1024        True       
     relu_77                 [512, 28, 28]          0           False      
     bottleneck_78           [512, 28, 28]          0           False      
     conv2d_79               [256, 28, 28]          131072      True       
     batchnorm2d_80          [256, 28, 28]          512         True       
     relu_81                 [256, 28, 28]          0           False      
     conv2d_82               [256, 14, 14]          589824      True       
     batchnorm2d_83          [256, 14, 14]          512         True       
     relu_84                 [256, 14, 14]          0           False      
     conv2d_85               [1024, 14, 14]         262144      True       
     batchnorm2d_86          [1024, 14, 14]         2048        True       
     conv2d_87               [1024, 14, 14]         524288      True       
     batchnorm2d_88          [1024, 14, 14]         2048        True       
     relu_89                 [1024, 14, 14]         0           False      
     bottleneck_90           [1024, 14, 14]         0           False      
     conv2d_91               [256, 14, 14]          262144      True       
     batchnorm2d_92          [256, 14, 14]          512         True       
     relu_93                 [256, 14, 14]          0           False      
     conv2d_94               [256, 14, 14]          589824      True       
     batchnorm2d_95          [256, 14, 14]          512         True       
     relu_96                 [256, 14, 14]          0           False      
     conv2d_97               [1024, 14, 14]         262144      True       
     batchnorm2d_98          [1024, 14, 14]         2048        True       
     relu_99                 [1024, 14, 14]         0           False      
     bottleneck_100          [1024, 14, 14]         0           False      
     conv2d_101              [256, 14, 14]          262144      True       
     batchnorm2d_102         [256, 14, 14]          512         True       
     relu_103                [256, 14, 14]          0           False      
     conv2d_104              [256, 14, 14]          589824      True       
     batchnorm2d_105         [256, 14, 14]          512         True       
     relu_106                [256, 14, 14]          0           False      
     conv2d_107              [1024, 14, 14]         262144      True       
     batchnorm2d_108         [1024, 14, 14]         2048        True       
     relu_109                [1024, 14, 14]         0           False      
     bottleneck_110          [1024, 14, 14]         0           False      
     conv2d_111              [256, 14, 14]          262144      True       
     batchnorm2d_112         [256, 14, 14]          512         True       
     relu_113                [256, 14, 14]          0           False      
     conv2d_114              [256, 14, 14]          589824      True       
     batchnorm2d_115         [256, 14, 14]          512         True       
     relu_116                [256, 14, 14]          0           False      
     conv2d_117              [1024, 14, 14]         262144      True       
     batchnorm2d_118         [1024, 14, 14]         2048        True       
     relu_119                [1024, 14, 14]         0           False      
     bottleneck_120          [1024, 14, 14]         0           False      
     conv2d_121              [256, 14, 14]          262144      True       
     batchnorm2d_122         [256, 14, 14]          512         True       
     relu_123                [256, 14, 14]          0           False      
     conv2d_124              [256, 14, 14]          589824      True       
     batchnorm2d_125         [256, 14, 14]          512         True       
     relu_126                [256, 14, 14]          0           False      
     conv2d_127              [1024, 14, 14]         262144      True       
     batchnorm2d_128         [1024, 14, 14]         2048        True       
     relu_129                [1024, 14, 14]         0           False      
     bottleneck_130          [1024, 14, 14]         0           False      
     conv2d_131              [256, 14, 14]          262144      True       
     batchnorm2d_132         [256, 14, 14]          512         True       
     relu_133                [256, 14, 14]          0           False      
     conv2d_134              [256, 14, 14]          589824      True       
     batchnorm2d_135         [256, 14, 14]          512         True       
     relu_136                [256, 14, 14]          0           False      
     conv2d_137              [1024, 14, 14]         262144      True       
     batchnorm2d_138         [1024, 14, 14]         2048        True       
     relu_139                [1024, 14, 14]         0           False      
     bottleneck_140          [1024, 14, 14]         0           False      
     conv2d_141              [512, 14, 14]          524288      True       
     batchnorm2d_142         [512, 14, 14]          1024        True       
     relu_143                [512, 14, 14]          0           False      
     conv2d_144              [512, 7, 7]            2359296     True       
     batchnorm2d_145         [512, 7, 7]            1024        True       
     relu_146                [512, 7, 7]            0           False      
     conv2d_147              [2048, 7, 7]           1048576     True       
     batchnorm2d_148         [2048, 7, 7]           4096        True       
     conv2d_149              [2048, 7, 7]           2097152     True       
     batchnorm2d_150         [2048, 7, 7]           4096        True       
     relu_151                [2048, 7, 7]           0           False      
     bottleneck_152          [2048, 7, 7]           0           False      
     conv2d_153              [512, 7, 7]            1048576     True       
     batchnorm2d_154         [512, 7, 7]            1024        True       
     relu_155                [512, 7, 7]            0           False      
     conv2d_156              [512, 7, 7]            2359296     True       
     batchnorm2d_157         [512, 7, 7]            1024        True       
     relu_158                [512, 7, 7]            0           False      
     conv2d_159              [2048, 7, 7]           1048576     True       
     batchnorm2d_160         [2048, 7, 7]           4096        True       
     relu_161                [2048, 7, 7]           0           False      
     bottleneck_162          [2048, 7, 7]           0           False      
     conv2d_163              [512, 7, 7]            1048576     True       
     batchnorm2d_164         [512, 7, 7]            1024        True       
     relu_165                [512, 7, 7]            0           False      
     conv2d_166              [512, 7, 7]            2359296     True       
     batchnorm2d_167         [512, 7, 7]            1024        True       
     relu_168                [512, 7, 7]            0           False      
     conv2d_169              [2048, 7, 7]           1048576     True       
     batchnorm2d_170         [2048, 7, 7]           4096        True       
     relu_171                [2048, 7, 7]           0           False      
     bottleneck_172          [2048, 7, 7]           0           False      
     adaptiveavgpool2d_173   [2048, 1, 1]           0           False      
     linear_174              [1000]                 2049000     True       
                                                                           
   Green layers are trainable layers, Cyan layers are non-trainable layers or frozen layers.
   Gray layers indicates this layer has been replaced by an Identity layer.
   Use to_embedding_model(...) to create embedding model.    
   ```
   ````
   ````{tab} Keras
   ```console
     name                  output_shape_display   nb_params   trainable  
    ──────────────────────────────────────────────────────────────────── 
     input_2               []                     0           False      
     conv1_pad             [230, 230, 3]          0           False      
     conv1_conv            [112, 112, 64]         9472        True       
     conv1_bn              [112, 112, 64]         256         True       
     conv1_relu            [112, 112, 64]         0           False      
     pool1_pad             [114, 114, 64]         0           False      
     pool1_pool            [56, 56, 64]           0           False      
     conv2_block1_1_conv   [56, 56, 64]           4160        True       
     conv2_block1_1_bn     [56, 56, 64]           256         True       
     conv2_block1_1_relu   [56, 56, 64]           0           False      
     conv2_block1_2_conv   [56, 56, 64]           36928       True       
     conv2_block1_2_bn     [56, 56, 64]           256         True       
     conv2_block1_2_relu   [56, 56, 64]           0           False      
     conv2_block1_0_conv   [56, 56, 256]          16640       True       
     conv2_block1_3_conv   [56, 56, 256]          16640       True       
     conv2_block1_0_bn     [56, 56, 256]          1024        True       
     conv2_block1_3_bn     [56, 56, 256]          1024        True       
     conv2_block1_add      [56, 56, 256]          0           False      
     conv2_block1_out      [56, 56, 256]          0           False      
     conv2_block2_1_conv   [56, 56, 64]           16448       True       
     conv2_block2_1_bn     [56, 56, 64]           256         True       
     conv2_block2_1_relu   [56, 56, 64]           0           False      
     conv2_block2_2_conv   [56, 56, 64]           36928       True       
     conv2_block2_2_bn     [56, 56, 64]           256         True       
     conv2_block2_2_relu   [56, 56, 64]           0           False      
     conv2_block2_3_conv   [56, 56, 256]          16640       True       
     conv2_block2_3_bn     [56, 56, 256]          1024        True       
     conv2_block2_add      [56, 56, 256]          0           False      
     conv2_block2_out      [56, 56, 256]          0           False      
     conv2_block3_1_conv   [56, 56, 64]           16448       True       
     conv2_block3_1_bn     [56, 56, 64]           256         True       
     conv2_block3_1_relu   [56, 56, 64]           0           False      
     conv2_block3_2_conv   [56, 56, 64]           36928       True       
     conv2_block3_2_bn     [56, 56, 64]           256         True       
     conv2_block3_2_relu   [56, 56, 64]           0           False      
     conv2_block3_3_conv   [56, 56, 256]          16640       True       
     conv2_block3_3_bn     [56, 56, 256]          1024        True       
     conv2_block3_add      [56, 56, 256]          0           False      
     conv2_block3_out      [56, 56, 256]          0           False      
     conv3_block1_1_conv   [28, 28, 128]          32896       True       
     conv3_block1_1_bn     [28, 28, 128]          512         True       
     conv3_block1_1_relu   [28, 28, 128]          0           False      
     conv3_block1_2_conv   [28, 28, 128]          147584      True       
     conv3_block1_2_bn     [28, 28, 128]          512         True       
     conv3_block1_2_relu   [28, 28, 128]          0           False      
     conv3_block1_0_conv   [28, 28, 512]          131584      True       
     conv3_block1_3_conv   [28, 28, 512]          66048       True       
     conv3_block1_0_bn     [28, 28, 512]          2048        True       
     conv3_block1_3_bn     [28, 28, 512]          2048        True       
     conv3_block1_add      [28, 28, 512]          0           False      
     conv3_block1_out      [28, 28, 512]          0           False      
     conv3_block2_1_conv   [28, 28, 128]          65664       True       
     conv3_block2_1_bn     [28, 28, 128]          512         True       
     conv3_block2_1_relu   [28, 28, 128]          0           False      
     conv3_block2_2_conv   [28, 28, 128]          147584      True       
     conv3_block2_2_bn     [28, 28, 128]          512         True       
     conv3_block2_2_relu   [28, 28, 128]          0           False      
     conv3_block2_3_conv   [28, 28, 512]          66048       True       
     conv3_block2_3_bn     [28, 28, 512]          2048        True       
     conv3_block2_add      [28, 28, 512]          0           False      
     conv3_block2_out      [28, 28, 512]          0           False      
     conv3_block3_1_conv   [28, 28, 128]          65664       True       
     conv3_block3_1_bn     [28, 28, 128]          512         True       
     conv3_block3_1_relu   [28, 28, 128]          0           False      
     conv3_block3_2_conv   [28, 28, 128]          147584      True       
     conv3_block3_2_bn     [28, 28, 128]          512         True       
     conv3_block3_2_relu   [28, 28, 128]          0           False      
     conv3_block3_3_conv   [28, 28, 512]          66048       True       
     conv3_block3_3_bn     [28, 28, 512]          2048        True       
     conv3_block3_add      [28, 28, 512]          0           False      
     conv3_block3_out      [28, 28, 512]          0           False      
     conv3_block4_1_conv   [28, 28, 128]          65664       True       
     conv3_block4_1_bn     [28, 28, 128]          512         True       
     conv3_block4_1_relu   [28, 28, 128]          0           False      
     conv3_block4_2_conv   [28, 28, 128]          147584      True       
     conv3_block4_2_bn     [28, 28, 128]          512         True       
     conv3_block4_2_relu   [28, 28, 128]          0           False      
     conv3_block4_3_conv   [28, 28, 512]          66048       True       
     conv3_block4_3_bn     [28, 28, 512]          2048        True       
     conv3_block4_add      [28, 28, 512]          0           False      
     conv3_block4_out      [28, 28, 512]          0           False      
     conv4_block1_1_conv   [14, 14, 256]          131328      True       
     conv4_block1_1_bn     [14, 14, 256]          1024        True       
     conv4_block1_1_relu   [14, 14, 256]          0           False      
     conv4_block1_2_conv   [14, 14, 256]          590080      True       
     conv4_block1_2_bn     [14, 14, 256]          1024        True       
     conv4_block1_2_relu   [14, 14, 256]          0           False      
     conv4_block1_0_conv   [14, 14, 1024]         525312      True       
     conv4_block1_3_conv   [14, 14, 1024]         263168      True       
     conv4_block1_0_bn     [14, 14, 1024]         4096        True       
     conv4_block1_3_bn     [14, 14, 1024]         4096        True       
     conv4_block1_add      [14, 14, 1024]         0           False      
     conv4_block1_out      [14, 14, 1024]         0           False      
     conv4_block2_1_conv   [14, 14, 256]          262400      True       
     conv4_block2_1_bn     [14, 14, 256]          1024        True       
     conv4_block2_1_relu   [14, 14, 256]          0           False      
     conv4_block2_2_conv   [14, 14, 256]          590080      True       
     conv4_block2_2_bn     [14, 14, 256]          1024        True       
     conv4_block2_2_relu   [14, 14, 256]          0           False      
     conv4_block2_3_conv   [14, 14, 1024]         263168      True       
     conv4_block2_3_bn     [14, 14, 1024]         4096        True       
     conv4_block2_add      [14, 14, 1024]         0           False      
     conv4_block2_out      [14, 14, 1024]         0           False      
     conv4_block3_1_conv   [14, 14, 256]          262400      True       
     conv4_block3_1_bn     [14, 14, 256]          1024        True       
     conv4_block3_1_relu   [14, 14, 256]          0           False      
     conv4_block3_2_conv   [14, 14, 256]          590080      True       
     conv4_block3_2_bn     [14, 14, 256]          1024        True       
     conv4_block3_2_relu   [14, 14, 256]          0           False      
     conv4_block3_3_conv   [14, 14, 1024]         263168      True       
     conv4_block3_3_bn     [14, 14, 1024]         4096        True       
     conv4_block3_add      [14, 14, 1024]         0           False      
     conv4_block3_out      [14, 14, 1024]         0           False      
     conv4_block4_1_conv   [14, 14, 256]          262400      True       
     conv4_block4_1_bn     [14, 14, 256]          1024        True       
     conv4_block4_1_relu   [14, 14, 256]          0           False      
     conv4_block4_2_conv   [14, 14, 256]          590080      True       
     conv4_block4_2_bn     [14, 14, 256]          1024        True       
     conv4_block4_2_relu   [14, 14, 256]          0           False      
     conv4_block4_3_conv   [14, 14, 1024]         263168      True       
     conv4_block4_3_bn     [14, 14, 1024]         4096        True       
     conv4_block4_add      [14, 14, 1024]         0           False      
     conv4_block4_out      [14, 14, 1024]         0           False      
     conv4_block5_1_conv   [14, 14, 256]          262400      True       
     conv4_block5_1_bn     [14, 14, 256]          1024        True       
     conv4_block5_1_relu   [14, 14, 256]          0           False      
     conv4_block5_2_conv   [14, 14, 256]          590080      True       
     conv4_block5_2_bn     [14, 14, 256]          1024        True       
     conv4_block5_2_relu   [14, 14, 256]          0           False      
     conv4_block5_3_conv   [14, 14, 1024]         263168      True       
     conv4_block5_3_bn     [14, 14, 1024]         4096        True       
     conv4_block5_add      [14, 14, 1024]         0           False      
     conv4_block5_out      [14, 14, 1024]         0           False      
     conv4_block6_1_conv   [14, 14, 256]          262400      True       
     conv4_block6_1_bn     [14, 14, 256]          1024        True       
     conv4_block6_1_relu   [14, 14, 256]          0           False      
     conv4_block6_2_conv   [14, 14, 256]          590080      True       
     conv4_block6_2_bn     [14, 14, 256]          1024        True       
     conv4_block6_2_relu   [14, 14, 256]          0           False      
     conv4_block6_3_conv   [14, 14, 1024]         263168      True       
     conv4_block6_3_bn     [14, 14, 1024]         4096        True       
     conv4_block6_add      [14, 14, 1024]         0           False      
     conv4_block6_out      [14, 14, 1024]         0           False      
     conv5_block1_1_conv   [7, 7, 512]            524800      True       
     conv5_block1_1_bn     [7, 7, 512]            2048        True       
     conv5_block1_1_relu   [7, 7, 512]            0           False      
     conv5_block1_2_conv   [7, 7, 512]            2359808     True       
     conv5_block1_2_bn     [7, 7, 512]            2048        True       
     conv5_block1_2_relu   [7, 7, 512]            0           False      
     conv5_block1_0_conv   [7, 7, 2048]           2099200     True       
     conv5_block1_3_conv   [7, 7, 2048]           1050624     True       
     conv5_block1_0_bn     [7, 7, 2048]           8192        True       
     conv5_block1_3_bn     [7, 7, 2048]           8192        True       
     conv5_block1_add      [7, 7, 2048]           0           False      
     conv5_block1_out      [7, 7, 2048]           0           False      
     conv5_block2_1_conv   [7, 7, 512]            1049088     True       
     conv5_block2_1_bn     [7, 7, 512]            2048        True       
     conv5_block2_1_relu   [7, 7, 512]            0           False      
     conv5_block2_2_conv   [7, 7, 512]            2359808     True       
     conv5_block2_2_bn     [7, 7, 512]            2048        True       
     conv5_block2_2_relu   [7, 7, 512]            0           False      
     conv5_block2_3_conv   [7, 7, 2048]           1050624     True       
     conv5_block2_3_bn     [7, 7, 2048]           8192        True       
     conv5_block2_add      [7, 7, 2048]           0           False      
     conv5_block2_out      [7, 7, 2048]           0           False      
     conv5_block3_1_conv   [7, 7, 512]            1049088     True       
     conv5_block3_1_bn     [7, 7, 512]            2048        True       
     conv5_block3_1_relu   [7, 7, 512]            0           False      
     conv5_block3_2_conv   [7, 7, 512]            2359808     True       
     conv5_block3_2_bn     [7, 7, 512]            2048        True       
     conv5_block3_2_relu   [7, 7, 512]            0           False      
     conv5_block3_3_conv   [7, 7, 2048]           1050624     True       
     conv5_block3_3_bn     [7, 7, 2048]           8192        True       
     conv5_block3_add      [7, 7, 2048]           0           False      
     conv5_block3_out      [7, 7, 2048]           0           False      
     avg_pool              [2048]                 0           False      
     predictions           [1000]                 2049000     True       
                                                                         
   Green layers are trainable layers, Cyan layers are non-trainable layers or frozen layers.
   Gray layers indicates this layer has been replaced by an Identity layer.
   Use to_embedding_model(...) to create embedding model.   
   ```   
   
   ````
   ````{tab} Paddle
   
   ```console
     name                    output_shape_display   nb_params   trainable  
    ────────────────────────────────────────────────────────────────────── 
     conv2d_1                [64, 112, 112]         9408        True       
     batchnorm2d_2           [64, 112, 112]         256         True       
     relu_3                  [64, 112, 112]         0           False      
     maxpool2d_4             [64, 56, 56]           0           False      
     conv2d_5                [64, 56, 56]           4096        True       
     batchnorm2d_6           [64, 56, 56]           256         True       
     relu_7                  [64, 56, 56]           0           False      
     conv2d_8                [64, 56, 56]           36864       True       
     batchnorm2d_9           [64, 56, 56]           256         True       
     relu_10                 [64, 56, 56]           0           False      
     conv2d_11               [256, 56, 56]          16384       True       
     batchnorm2d_12          [256, 56, 56]          1024        True       
     conv2d_13               [256, 56, 56]          16384       True       
     batchnorm2d_14          [256, 56, 56]          1024        True       
     relu_15                 [256, 56, 56]          0           False      
     bottleneckblock_16      [256, 56, 56]          0           False      
     conv2d_17               [64, 56, 56]           16384       True       
     batchnorm2d_18          [64, 56, 56]           256         True       
     relu_19                 [64, 56, 56]           0           False      
     conv2d_20               [64, 56, 56]           36864       True       
     batchnorm2d_21          [64, 56, 56]           256         True       
     relu_22                 [64, 56, 56]           0           False      
     conv2d_23               [256, 56, 56]          16384       True       
     batchnorm2d_24          [256, 56, 56]          1024        True       
     relu_25                 [256, 56, 56]          0           False      
     bottleneckblock_26      [256, 56, 56]          0           False      
     conv2d_27               [64, 56, 56]           16384       True       
     batchnorm2d_28          [64, 56, 56]           256         True       
     relu_29                 [64, 56, 56]           0           False      
     conv2d_30               [64, 56, 56]           36864       True       
     batchnorm2d_31          [64, 56, 56]           256         True       
     relu_32                 [64, 56, 56]           0           False      
     conv2d_33               [256, 56, 56]          16384       True       
     batchnorm2d_34          [256, 56, 56]          1024        True       
     relu_35                 [256, 56, 56]          0           False      
     bottleneckblock_36      [256, 56, 56]          0           False      
     conv2d_37               [128, 56, 56]          32768       True       
     batchnorm2d_38          [128, 56, 56]          512         True       
     relu_39                 [128, 56, 56]          0           False      
     conv2d_40               [128, 28, 28]          147456      True       
     batchnorm2d_41          [128, 28, 28]          512         True       
     relu_42                 [128, 28, 28]          0           False      
     conv2d_43               [512, 28, 28]          65536       True       
     batchnorm2d_44          [512, 28, 28]          2048        True       
     conv2d_45               [512, 28, 28]          131072      True       
     batchnorm2d_46          [512, 28, 28]          2048        True       
     relu_47                 [512, 28, 28]          0           False      
     bottleneckblock_48      [512, 28, 28]          0           False      
     conv2d_49               [128, 28, 28]          65536       True       
     batchnorm2d_50          [128, 28, 28]          512         True       
     relu_51                 [128, 28, 28]          0           False      
     conv2d_52               [128, 28, 28]          147456      True       
     batchnorm2d_53          [128, 28, 28]          512         True       
     relu_54                 [128, 28, 28]          0           False      
     conv2d_55               [512, 28, 28]          65536       True       
     batchnorm2d_56          [512, 28, 28]          2048        True       
     relu_57                 [512, 28, 28]          0           False      
     bottleneckblock_58      [512, 28, 28]          0           False      
     conv2d_59               [128, 28, 28]          65536       True       
     batchnorm2d_60          [128, 28, 28]          512         True       
     relu_61                 [128, 28, 28]          0           False      
     conv2d_62               [128, 28, 28]          147456      True       
     batchnorm2d_63          [128, 28, 28]          512         True       
     relu_64                 [128, 28, 28]          0           False      
     conv2d_65               [512, 28, 28]          65536       True       
     batchnorm2d_66          [512, 28, 28]          2048        True       
     relu_67                 [512, 28, 28]          0           False      
     bottleneckblock_68      [512, 28, 28]          0           False      
     conv2d_69               [128, 28, 28]          65536       True       
     batchnorm2d_70          [128, 28, 28]          512         True       
     relu_71                 [128, 28, 28]          0           False      
     conv2d_72               [128, 28, 28]          147456      True       
     batchnorm2d_73          [128, 28, 28]          512         True       
     relu_74                 [128, 28, 28]          0           False      
     conv2d_75               [512, 28, 28]          65536       True       
     batchnorm2d_76          [512, 28, 28]          2048        True       
     relu_77                 [512, 28, 28]          0           False      
     bottleneckblock_78      [512, 28, 28]          0           False      
     conv2d_79               [256, 28, 28]          131072      True       
     batchnorm2d_80          [256, 28, 28]          1024        True       
     relu_81                 [256, 28, 28]          0           False      
     conv2d_82               [256, 14, 14]          589824      True       
     batchnorm2d_83          [256, 14, 14]          1024        True       
     relu_84                 [256, 14, 14]          0           False      
     conv2d_85               [1024, 14, 14]         262144      True       
     batchnorm2d_86          [1024, 14, 14]         4096        True       
     conv2d_87               [1024, 14, 14]         524288      True       
     batchnorm2d_88          [1024, 14, 14]         4096        True       
     relu_89                 [1024, 14, 14]         0           False      
     bottleneckblock_90      [1024, 14, 14]         0           False      
     conv2d_91               [256, 14, 14]          262144      True       
     batchnorm2d_92          [256, 14, 14]          1024        True       
     relu_93                 [256, 14, 14]          0           False      
     conv2d_94               [256, 14, 14]          589824      True       
     batchnorm2d_95          [256, 14, 14]          1024        True       
     relu_96                 [256, 14, 14]          0           False      
     conv2d_97               [1024, 14, 14]         262144      True       
     batchnorm2d_98          [1024, 14, 14]         4096        True       
     relu_99                 [1024, 14, 14]         0           False      
     bottleneckblock_100     [1024, 14, 14]         0           False      
     conv2d_101              [256, 14, 14]          262144      True       
     batchnorm2d_102         [256, 14, 14]          1024        True       
     relu_103                [256, 14, 14]          0           False      
     conv2d_104              [256, 14, 14]          589824      True       
     batchnorm2d_105         [256, 14, 14]          1024        True       
     relu_106                [256, 14, 14]          0           False      
     conv2d_107              [1024, 14, 14]         262144      True       
     batchnorm2d_108         [1024, 14, 14]         4096        True       
     relu_109                [1024, 14, 14]         0           False      
     bottleneckblock_110     [1024, 14, 14]         0           False      
     conv2d_111              [256, 14, 14]          262144      True       
     batchnorm2d_112         [256, 14, 14]          1024        True       
     relu_113                [256, 14, 14]          0           False      
     conv2d_114              [256, 14, 14]          589824      True       
     batchnorm2d_115         [256, 14, 14]          1024        True       
     relu_116                [256, 14, 14]          0           False      
     conv2d_117              [1024, 14, 14]         262144      True       
     batchnorm2d_118         [1024, 14, 14]         4096        True       
     relu_119                [1024, 14, 14]         0           False      
     bottleneckblock_120     [1024, 14, 14]         0           False      
     conv2d_121              [256, 14, 14]          262144      True       
     batchnorm2d_122         [256, 14, 14]          1024        True       
     relu_123                [256, 14, 14]          0           False      
     conv2d_124              [256, 14, 14]          589824      True       
     batchnorm2d_125         [256, 14, 14]          1024        True       
     relu_126                [256, 14, 14]          0           False      
     conv2d_127              [1024, 14, 14]         262144      True       
     batchnorm2d_128         [1024, 14, 14]         4096        True       
     relu_129                [1024, 14, 14]         0           False      
     bottleneckblock_130     [1024, 14, 14]         0           False      
     conv2d_131              [256, 14, 14]          262144      True       
     batchnorm2d_132         [256, 14, 14]          1024        True       
     relu_133                [256, 14, 14]          0           False      
     conv2d_134              [256, 14, 14]          589824      True       
     batchnorm2d_135         [256, 14, 14]          1024        True       
     relu_136                [256, 14, 14]          0           False      
     conv2d_137              [1024, 14, 14]         262144      True       
     batchnorm2d_138         [1024, 14, 14]         4096        True       
     relu_139                [1024, 14, 14]         0           False      
     bottleneckblock_140     [1024, 14, 14]         0           False      
     conv2d_141              [512, 14, 14]          524288      True       
     batchnorm2d_142         [512, 14, 14]          2048        True       
     relu_143                [512, 14, 14]          0           False      
     conv2d_144              [512, 7, 7]            2359296     True       
     batchnorm2d_145         [512, 7, 7]            2048        True       
     relu_146                [512, 7, 7]            0           False      
     conv2d_147              [2048, 7, 7]           1048576     True       
     batchnorm2d_148         [2048, 7, 7]           8192        True       
     conv2d_149              [2048, 7, 7]           2097152     True       
     batchnorm2d_150         [2048, 7, 7]           8192        True       
     relu_151                [2048, 7, 7]           0           False      
     bottleneckblock_152     [2048, 7, 7]           0           False      
     conv2d_153              [512, 7, 7]            1048576     True       
     batchnorm2d_154         [512, 7, 7]            2048        True       
     relu_155                [512, 7, 7]            0           False      
     conv2d_156              [512, 7, 7]            2359296     True       
     batchnorm2d_157         [512, 7, 7]            2048        True       
     relu_158                [512, 7, 7]            0           False      
     conv2d_159              [2048, 7, 7]           1048576     True       
     batchnorm2d_160         [2048, 7, 7]           8192        True       
     relu_161                [2048, 7, 7]           0           False      
     bottleneckblock_162     [2048, 7, 7]           0           False      
     conv2d_163              [512, 7, 7]            1048576     True       
     batchnorm2d_164         [512, 7, 7]            2048        True       
     relu_165                [512, 7, 7]            0           False      
     conv2d_166              [512, 7, 7]            2359296     True       
     batchnorm2d_167         [512, 7, 7]            2048        True       
     relu_168                [512, 7, 7]            0           False      
     conv2d_169              [2048, 7, 7]           1048576     True       
     batchnorm2d_170         [2048, 7, 7]           8192        True       
     relu_171                [2048, 7, 7]           0           False      
     bottleneckblock_172     [2048, 7, 7]           0           False      
     adaptiveavgpool2d_173   [2048, 1, 1]           0           False      
     linear_174              [1000]                 2049000     True       
                                                                           
   Green layers are trainable layers, Cyan layers are non-trainable layers or frozen layers.
   Gray layers indicates this layer has been replaced by an Identity layer.
   Use to_embedding_model(...) to create embedding model.    
   ```      
   ````

## Select an embedding layer as your model output.

After plotted the table summary of the pre-trained model, we can decide which layer can be used as the "embedding layer".
As you can see in the above ResNet-50, layer with name ``linear_174`` (pytorch/paddle) or ``predictions`` (keras) is the final classification layer for classify 1000 ImageNet classes.
This layer is not a good chocie of "embedding layer". We'll use ``adaptiveavgpool2d_173`` (pytorch/paddle) or ``avg_pool`` (keras) as our "embedding layer".

Keep the layer name in mind, we'll put everything together after we decide ``freeze`` strategy and ``bottleneck layer``.

## Decide your `freeze` strategy.

To apply a pre-trained model on your downstream task, in most cases you do not want to train everything from scratch.
Finetuner allows you to freeze the entire model or freeze specific layers, with ``freeze`` argument.

If ``freeze=True``, finetuner will freeze the entire pre-trained model. If freeze is a list of string layer names,
for example, ``freeze=['conv2d_1', 'conv2d_5']`` will only freeze these two layers.

Again, keep this in mind, we'll put everything together after we attach the ``bottleneck layer``.

## Attach a bottleneck module.

Sometimes you want to add a bottleneck module or projection head on top of your embedding model.
This bottleneck layer should be a simple multi-layer perceptron.
This could help you improve your embedding quality and potentially reduce the dimensionality.

Tailor allows you to attach this small bottleneck module on top of your embedding model.
In the below example, we put everything together, including:

- choose embedding layer
- freezing weights of specific layers.
- attach a bottleneck layer

with the method called ``to_embedding_model`` method.

1. Tailor provides a high-level API ``finetuner.tailor.to_embedding_model``, which can be used as follows:
    ````{tab} PyTorch
    ```python
      import torch.nn as nn
      import torchvision
      import finetuner as ft
   
      model = torchvision.models.resnet50(pretrained=True)
   
      class SimpleMLP(nn.Module):
          def __init__(self):
              super().__init__()
              self.l1 = nn.Linear(2048, 2048, bias=True)
              self.relu = nn.ReLU()
              self.l2 = nn.Linear(2048, 1024, bias=True)
          def forward(self, x):
              return self.l2((self.relu(self.l1(x))))
   
      new_model = ft.tailor.to_embedding_model(
          model=model,
          layer_name='adaptiveavgpool2d_173',
          freeze=['conv2d_1', 'batchnorm2d_2'],  # or set to True to freeze the entire model
          bottleneck_net=SimpleMLP(),
          input_size=(3, 224, 224),
      )
    ```

    ````
    ````{tab} Keras
    ```python
    import tensorflow as tf
    import finetuner as ft
   
    model = tf.keras.applications.ResNet50(weights='imagenet')

    bottleneck_model = tf.keras.models.Sequential()
    bottleneck_model.add(tf.keras.layers.InputLayer(input_shape=(2048,)))
    bottleneck_model.add(tf.keras.layers.Dense(2048, activation='relu'))
    bottleneck_model.add(tf.keras.layers.Dense(1024))

    new_model = ft.tailor.to_embedding_model(
        model=model,
        layer_name='avg_pool',
        freeze=['conv1_conv', 'conv1_bn'],  # or set to True to freeze the entire model
        bottleneck_net=bottleneck_model,
        input_size=(3, 224, 224),
    )
    ```
    ````
    ````{tab} Paddle
    ```python
    import paddle
    import paddle.nn as nn
    import finetuner as ft
   
    model = paddle.vision.models.resnet50(pretrained=True)

    class SimpleMLP(nn.Layer):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(2048, 2048)
            self.relu = nn.ReLU()
            self.l2 = nn.Linear(2048, 1024)
   
          def forward(self, x):
              return self.l2(self.relu(self.l1(x)))
        
    new_model = ft.tailor.to_embedding_model(
        model=model,
        layer_name='adaptiveavgpool2d_173',
        freeze=['conv2d_1', 'batchnorm2d_2'],  # or set to True to freeze the entire model
        bottleneck_net=SimpleMLP(),
        input_size=(3, 224, 224),
    )
    ```
    ````

if you ``display`` the `new_model`, you'll notice that the `fc` layer has been replaced with as `Identity`.
The layers you specified to `freeze` are not trainable anymore.
Last but not least, you attached a trainable bottleneck module and reduced the ouput dimensionality to `1024`.


## Tips

- For PyTorch/Paddle models, having the correct `input_size` and `input_dtype` is fundamental to use `to_embedding_model` and `display`.
- You can chop off layers and concat new layers afterward. To get the accurate layer name, you can first use `display` to list all layers. 
- Different frameworks may give different layer names. Often, PyTorch and Paddle layer names are consistent.
