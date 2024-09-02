# -*- encoding: utf-8 -*-
"""
@File    :   ema.py
@Time    :   2024/08/28 21:16:44
@Author  :   lihao57
@Version :   1.0
@Contact :   lihao57@baidu.com
"""


import paddle
from copy import deepcopy


class ExponentialMovingAverage:
    """Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    E.g. Google's hyper-params for training MNASNet, MobileNet-V3, EfficientNet, etc that use
    RMSprop with a short 2.4-3 epoch decay period and slow LR decay rate of .96-.99 requires EMA
    smoothing of weights to match results. Pay attention to the decay constant you are using
    relative to your update count per epoch.
    To keep EMA from using GPU resources, set device='cpu'. This will save a bit of memory but
    disable validation of the EMA weights. Validation will have to be done manually in a separate
    process, or after the training stops converging.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    I've tested with the sequence in my own train.py for torch.DataParallel, apex.DDP, and single-GPU.
    """

    def __init__(self, model, decay=0.999):
        # make a copy of the model for accumulating moving average of weights
        self.model = model
        self.ema = deepcopy(model)
        self.ema.eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.stop_gradient = True

    def update(self, model):
        """update"""
        self.model = model
        decay = self.decay
        with paddle.no_grad():
            msd, esd = model.state_dict(), self.ema.state_dict()
            for k, v in msd.items():
                # 只通过EMA更细可学习参数
                if v.stop_gradient:
                    esd[k] = v.detach()
                else:
                    esd[k] = decay * esd[k] + (1.0 - decay) * v.detach()
            self.ema.set_state_dict(esd)

    def get_model(self):
        """get model"""
        return self.model

    def get_ema_model(self):
        """get ema model"""
        return self.ema
