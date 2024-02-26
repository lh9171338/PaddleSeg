# -*- encoding: utf-8 -*-
"""
@File    :   wrap_functions.py
@Time    :   2024/02/24 19:17:34
@Author  :   lihao57
@Version :   1.0
@Contact :   lihao57@baidu.com
"""


import paddle
import paddle.nn as nn

"""
Warp the functon api, so the normal and quantization training can use the same network.
"""


class Add(nn.Layer):
    """
    Add
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, y, name=None):
        return paddle.add(x, y, name)


class Subtract(nn.Layer):
    """
    Subtract
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, y, name=None):
        return paddle.subtract(x, y, name)


class Multiply(nn.Layer):
    """
    Multiply
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, y, name=None):
        return paddle.multiply(x, y, name)


class Divide(nn.Layer):
    """
    Divide
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, y, name=None):
        return paddle.divide(x, y, name)


class Reshape(nn.Layer):
    """
    Reshape
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, shape, name=None):
        return paddle.reshape(x, shape, name)


class Transpose(nn.Layer):
    """
    Transpose
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, perm, name=None):
        return paddle.transpose(x, perm, name)


class Concat(nn.Layer):
    """
    Concat
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, axis=0, name=None):
        return paddle.concat(x, axis, name)


class Flatten(nn.Layer):
    """
    Flatten
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, start_axis=0, stop_axis=-1, name=None):
        return paddle.flatten(x, start_axis, stop_axis, name)
