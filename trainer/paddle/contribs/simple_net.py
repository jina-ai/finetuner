import paddle
import paddle.nn.functional as F
import numpy as np
import random

class SimpleNet(paddle.nn.Layer):
    def __init__(self):
        super(SimpleNet, self).__init__()

        self.conv1 = paddle.nn.Conv2D(in_channels=1,
                                      out_channels=32,
                                      kernel_size=(3, 3),
                                      stride=2)

        self.conv2 = paddle.nn.Conv2D(in_channels=32,
                                      out_channels=64,
                                      kernel_size=(3, 3),
                                      stride=2)

        self.conv3 = paddle.nn.Conv2D(in_channels=64,
                                      out_channels=128,
                                      kernel_size=(3, 3),
                                      stride=2)

        self.gloabl_pool = paddle.nn.AdaptiveAvgPool2D((1, 1))

        self.fc1 = paddle.nn.Linear(in_features=128, out_features=8)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.gloabl_pool(x)
        x = paddle.squeeze(x, axis=[2, 3])
        x = self.fc1(x)
        x = x / paddle.norm(x, axis=1, keepdim=True)
        return x