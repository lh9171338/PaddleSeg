# -*- encoding: utf-8 -*-
"""
@File    :   loss.py
@Time    :   2023/12/18 19:28:38
@Author  :   lihao57
@Version :   1.0
@Contact :   lihao57@baidu.com
"""


import paddle
import paddle.nn.functional as F
from ppseg.apis import manager
from ppseg.losses.utils import weighted_loss


__all__ = [
    "Loss",
    "MixedLoss",
    "BCELoss",
    "CELoss",
    "FocalLoss",
    "DiceLoss",
]


class Loss:
    """
    Loss
    """

    def __init__(
        self,
        weight=None,
        **kwargs,
    ):
        if isinstance(weight, (list, tuple)):
            weight = paddle.to_tensor(weight)

        self.weight = weight

    def __call__(self, pred, target):
        raise NotImplementedError


@manager.LOSSES.add_component
class MixedLoss:
    """
    Mixed Loss
    """

    def __init__(
        self,
        losses,
        weights,
    ):
        assert len(losses) == len(
            weights
        ), "The length of `losses` and `weights` must be the same"
        self.losses = losses
        self.weights = weights

    def __call__(self, pred, target):
        total_loss = 0
        for loss_func, weight in zip(self.losses, self.weights):
            loss = loss_func(pred, target)
            total_loss += loss * weight

        return total_loss


@manager.LOSSES.add_component
class BCELoss(Loss):
    """
    Binary Coss Entropy Loss
    """

    def __init__(
        self,
        with_logits=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.with_logits = with_logits

    @weighted_loss
    def __call__(self, pred, target):
        if self.with_logits:
            loss = F.binary_cross_entropy_with_logits(
                pred, target, reduction="none"
            )
        else:
            loss = F.binary_cross_entropy(pred, target, reduction="none")

        if self.weight is not None:
            loss = loss * self.weight

        return loss


@manager.LOSSES.add_component
class CELoss(Loss):
    """
    Coss Entropy Loss
    """

    def __init__(
        self,
        use_softmax=False,
        soft_label=False,
        ignore_index=-1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.use_softmax = use_softmax
        self.soft_label = soft_label
        self.ignore_index = ignore_index

    @weighted_loss
    def __call__(self, pred, target):
        """forward function.
        Args:
            pred (Tensor): logits of class prediction, of shape (N, H, W, C)
            target (Tensor): target class label, of shape (N, H, W)

        Returns:
            loss (Tensor): loss
        """
        num_classes = pred.shape[-1]
        if self.weight is not None:
            weight = self.weight
            if not isinstance(weight, paddle.Tensor):
                weight = paddle.to_tensor([weight] * num_classes)
        else:
            weight = None

        loss = F.cross_entropy(
            pred,
            target,
            ignore_index=self.ignore_index,
            soft_label=self.soft_label,
            use_softmax=self.use_softmax,
            weight=weight,
            reduction="none",
        )

        return loss


@manager.LOSSES.add_component
class FocalLoss(Loss):
    """A wrapper around paddle.nn.functional.sigmoid_focal_loss.
    Args:
        use_sigmoid (bool): currently only support use_sigmoid=True
        alpha (float): parameter alpha in Focal Loss
        gamma (float): parameter gamma in Focal Loss
        loss_weight (float): final loss will be multiplied by this
    """

    def __init__(
        self,
        use_sigmoid=True,
        alpha=0.25,
        gamma=2.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert (
            use_sigmoid == True
        ), "Focal Loss only supports sigmoid at the moment"
        self.use_sigmoid = use_sigmoid
        self.alpha = alpha
        self.gamma = gamma

    @weighted_loss
    def __call__(self, pred, target):
        """forward function.
        Args:
            pred (Tensor): logits of class prediction, of shape (N, H, W, C)
            target (Tensor): target class label, of shape (N, H, W)

        Returns:
            loss (Tensor): loss
        """
        num_classes = pred.shape[-1]
        if num_classes == 1:
            target = target.astype(pred.dtype)
        else:
            target = F.one_hot(target, num_classes)

        if len(target.shape) + 1 == len(pred.shape):
            target = target.unsqueeze(axis=-1)

        loss = F.sigmoid_focal_loss(
            pred, target, alpha=self.alpha, gamma=self.gamma, reduction="none"
        )
        if self.weight is not None:
            loss = loss * self.weight

        return loss


@manager.LOSSES.add_component
class DiceLoss(Loss):
    """
    Dice Loss
    """

    def __init__(
        self,
        smooth=1.0,
        separate=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.smooth = smooth
        self.separate = separate

    @weighted_loss
    def __call__(self, pred, target):
        """forward function.
        Args:
            pred (Tensor): logits of class prediction, of shape (N, H, W, C)
            target (Tensor): target class label, of shape (N, H, W)

        Returns:
            loss (Tensor): loss
        """
        if self.weight is not None:
            weight = paddle.to_tensor(self.weight)
        else:
            weight = None

        num_classes = pred.shape[-1]
        if num_classes == 1:
            pred = pred.sigmoid().squeeze(axis=-1)
            target = target.astype(pred.dtype)
        else:
            pred = F.softmax(pred, axis=-1)
            target = F.one_hot(target, num_classes)

        if self.separate:
            pred = pred.flatten(1, 2)
            target = target.flatten(1, 2)
            intersection = paddle.sum(pred * target, axis=1)
            cardinality = paddle.sum(pred + target, axis=1)
            loss = 1 - (2 * intersection + self.smooth) / (
                cardinality + self.smooth
            )

            if self.weight is not None:
                loss = loss * self.weight
        else:
            pred = pred.flatten(1)
            target = target.flatten(1)
            intersection = paddle.sum(pred * target, axis=1)
            cardinality = paddle.sum(pred + target, axis=1)
            loss = 1 - (2 * intersection + self.smooth) / (
                cardinality + self.smooth
            )

            if self.weight is not None:
                loss = loss * self.weight

        return loss
