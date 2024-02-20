# -*- encoding: utf-8 -*-
"""
@File    :   seg_head.py
@Time    :   2024/02/18 17:50:41
@Author  :   lihao57
@Version :   1.0
@Contact :   lihao57@baidu.com
"""


from cmath import isnan
from multiprocessing import reduction
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppseg.apis import manager


__all__ = ["SegHead"]


@manager.HEADS.add_component
class SegHead(nn.Layer):
    """
    SegHead for segmentation network
    """

    def __init__(
        self,
        loss_cls=None,
    ):
        super().__init__()

        self.loss_cls = loss_cls

    def forward(self, feat, **kwargs) -> dict:
        if isinstance(feat, (list, tuple)):
            feat = feat[-1]

        pred_dict = dict(
            outputs=feat,
        )

        return pred_dict

    def loss(self, pred_dict, label, **kwargs) -> dict:
        """
        loss

        Args:
            pred_dict (dict): dict of predicted outputs
            label (Tensor): label, shape [B, H, W]

        Returns:
            loss_dict (dict): dict of loss
        """
        pred = pred_dict["outputs"].transpose([0, 2, 3, 1])
        label = label.astype("int32")
        loss_dict = dict()

        # cls loss
        loss_cls = self.loss_cls(pred, label, reduction="none")
        bg_loss = loss_cls[label == 0].mean()
        fg_loss = loss_cls[label > 0].mean()
        bg_loss = paddle.nan_to_num(bg_loss, 0)
        fg_loss = paddle.nan_to_num(fg_loss, 0)
        loss_dict["bg_loss"] = bg_loss
        loss_dict["fg_loss"] = fg_loss

        return loss_dict

    def predict(self, pred_dict: dict, **kwargs) -> list:
        """predict"""
        pred = pred_dict["outputs"]
        pred = F.softmax(pred, axis=1)
        pred_label = paddle.argmax(pred, axis=1)
        pred_score = paddle.take_along_axis(
            pred, pred_label.unsqueeze(axis=1), axis=1
        ).squeeze(axis=1)

        B = pred.shape[0]
        results = []
        for batch_id in range(B):
            result = dict(
                pred_label=pred_label[batch_id],
                pred_score=pred_score[batch_id],
            )
            results.append(result)

        return results


if __name__ == "__main__":
    head = SegHead()
    feats = [
        paddle.randn([4, 2, 224, 224]),
    ]
    pred = head(feats)["outputs"]
    print(pred.shape)
