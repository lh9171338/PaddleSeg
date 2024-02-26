# -*- encoding: utf-8 -*-
"""
@File    :   unet.py
@Time    :   2024/02/18 17:31:52
@Author  :   lihao57
@Version :   1.0
@Contact :   lihao57@baidu.com
"""


import logging
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppseg.apis import manager
from ppseg.architectures import Segmentor
from ppseg import layers


__all__ = ["UNet"]


@manager.MODELS.add_component
class UNet(Segmentor):
    """
    The UNet implementation based on PaddlePaddle.

    The original article refers to
    Olaf Ronneberger, et, al. "U-Net: Convolutional Networks for Biomedical Image Segmentation"
    (https://arxiv.org/abs/1505.04597).

    Args:
        num_classes (int): The unique number of target classes.
        align_corners (bool): An argument of F.interpolate. It should be set to False when the output size of feature
            is even, e.g. 1024x512, otherwise it is True, e.g. 769x769.  Default: False.
        use_deconv (bool, optional): A bool value indicates whether using deconvolution in upsampling.
            If False, use resize_bilinear. Default: False.
        in_channels (int, optional): The channels of input image. Default: 3.
        pretrained (str, optional): The path or url of pretrained model for fine tuning. Default: None.
    """

    def __init__(
        self,
        num_classes,
        align_corners=False,
        use_deconv=False,
        in_channels=3,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.encode = Encoder(in_channels)
        self.decode = Decoder(align_corners, use_deconv=use_deconv)
        self.cls = self.conv = nn.Conv2D(
            in_channels=64,
            out_channels=num_classes,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.init_weight()

    def forward(self, x):
        x, short_cuts = self.encode(x)
        x = self.decode(x, short_cuts)
        logit = self.cls(x)
        logits = [logit]
        return logits


class Encoder(nn.Layer):
    """
    Encoder of UNet
    """

    def __init__(self, in_channels=3):
        super().__init__()

        self.double_conv = nn.Sequential(
            layers.ConvBNReLU(in_channels, 64, 3), layers.ConvBNReLU(64, 64, 3)
        )
        down_channels = [[64, 128], [128, 256], [256, 512], [512, 512]]
        self.down_sample_list = nn.LayerList(
            [
                self.down_sampling(channel[0], channel[1])
                for channel in down_channels
            ]
        )

    def down_sampling(self, in_channels, out_channels):
        """down sampling"""
        modules = []
        modules.append(nn.MaxPool2D(kernel_size=2, stride=2))
        modules.append(layers.ConvBNReLU(in_channels, out_channels, 3))
        modules.append(layers.ConvBNReLU(out_channels, out_channels, 3))
        return nn.Sequential(*modules)

    def forward(self, x):
        short_cuts = []
        x = self.double_conv(x)
        for down_sample in self.down_sample_list:
            short_cuts.append(x)
            x = down_sample(x)
        return x, short_cuts


class Decoder(nn.Layer):
    """
    Decoder of UNet
    """

    def __init__(self, align_corners, use_deconv=False):
        super().__init__()

        up_channels = [[512, 256], [256, 128], [128, 64], [64, 64]]
        self.up_sample_list = nn.LayerList(
            [
                UpSampling(channel[0], channel[1], align_corners, use_deconv)
                for channel in up_channels
            ]
        )

    def forward(self, x, short_cuts):
        for i in range(len(short_cuts)):
            x = self.up_sample_list[i](x, short_cuts[-(i + 1)])
        return x


class UpSampling(nn.Layer):
    """
    UpSampling of UNet
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        align_corners,
        use_deconv=False,
    ):
        super().__init__()

        self.align_corners = align_corners

        self.use_deconv = use_deconv
        if self.use_deconv:
            self.deconv = nn.Conv2DTranspose(
                in_channels,
                out_channels // 2,
                kernel_size=2,
                stride=2,
                padding=0,
            )
            in_channels = in_channels + out_channels // 2
        else:
            in_channels *= 2

        self.double_conv = nn.Sequential(
            layers.ConvBNReLU(in_channels, out_channels, 3),
            layers.ConvBNReLU(out_channels, out_channels, 3),
        )

    def forward(self, x, short_cut):
        if self.use_deconv:
            x = self.deconv(x)
        else:
            x = F.interpolate(
                x,
                paddle.shape(short_cut)[2:],
                mode="bilinear",
                align_corners=self.align_corners,
            )
        x = paddle.concat([x, short_cut], axis=1)
        x = self.double_conv(x)
        return x


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    net = UNet(
        num_classes=2,
        use_deconv=False,
        pretrained="https://bj.bcebos.com/paddleseg/dygraph/hrf/unet_hrf_256x256_40k/model.pdparams",
    )
    x = paddle.randn([4, 3, 224, 224])
    outs = net(x)
    for out in outs:
        print(out.shape)
