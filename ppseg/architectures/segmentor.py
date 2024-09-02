# -*- encoding: utf-8 -*-
"""
@File    :   yolo_detector.py
@Time    :   2023/12/16 23:10:44
@Author  :   lihao57
@Version :   1.0
@Contact :   lihao57@baidu.com
"""


import logging
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppseg.apis import manager
from ppseg.utils import utils


@manager.MODELS.add_component
class Segmentor(nn.Layer):
    """
    Segmentor
    """

    def __init__(
        self,
        pretrained=None,
        custom_lr_factor=None,
        loss=None,
        use_sigmoid=False,
        **kwargs,
    ):
        super().__init__()

        self.pretrained = pretrained
        self.custom_lr_factor = custom_lr_factor
        self.loss = loss
        self.use_sigmoid = use_sigmoid

    def init_weight(self):
        """initialize weights"""
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)

        if self.custom_lr_factor is not None:
            assert isinstance(self.custom_lr_factor, dict)
            for layer_name, lr_factor in self.custom_lr_factor.items():
                layer = getattr(self, layer_name, None)
                if layer:
                    logging.info(
                        "Set lr_factor={} in layer {}".format(
                            lr_factor, layer_name
                        )
                    )
                    for p in layer.parameters():
                        p.optimize_attr["learning_rate"] = lr_factor
                else:
                    logging.warning(
                        "Setting custom lr: layer {} is not in the model".format(
                            layer_name
                        )
                    )

    def forward(self, x) -> dict:
        """foward function"""
        raise NotImplementedError()

    def get_loss(self, logits, label, **kwargs) -> dict:
        """
        get loss

        Args:
            logits: list[Tensor], with shape [N, C, H, W]
            label: Tensor, with shape [B, H, W]

        Returns:
            loss_dict: dict of loss
        """
        if not isinstance(logits, (list, tuple)):
            logits = [logits]
        loss = paddle.to_tensor([0.0])
        label = label.astype("int32")
        for logit in logits:
            logit = logit.transpose([0, 2, 3, 1])
            loss += self.loss(logit, label)

        loss_dict = dict(loss=loss)

        return loss_dict

    def predict(self, logits, **kwargs) -> list:
        """
        predict

        Args:
            logits: list[Tensor], with shape [N, C, H, W]

        Returns:
            results: list of dict
        """
        if isinstance(logits, (list, tuple)):
            logit = logits[-1]
        else:
            logit = logits

        if logit.shape[1] == 1:
            pred_score = logit.sigmoid()
            pred_label = (pred_score >= 0.5).astype("int64")
        else:
            if self.use_sigmoid:
                pred = logit.sigmoid()
            else:
                pred = F.softmax(logit, axis=1)
            pred_label = paddle.argmax(pred, axis=1)
            pred_score = paddle.take_along_axis(
                pred, pred_label.unsqueeze(axis=1), axis=1
            ).squeeze(axis=1)

        B = logit.shape[0]
        results = []
        for batch_id in range(B):
            result = dict(
                pred_label=pred_label[batch_id],
                pred_score=pred_score[batch_id],
            )
            results.append(result)

        return results
