# -*- encoding: utf-8 -*-
"""
@File    :   seg_metric.py
@Time    :   2024/02/19 14:59:12
@Author  :   lihao57
@Version :   1.0
@Contact :   lihao57@baidu.com
"""


import os
import numpy as np
import paddle
import tqdm
from paddle.metric import Metric
from ppseg.apis import manager
import ppseg.apis.utils as api_utils


__all__ = ["MIoUMetric"]


@manager.METRICS.add_component
class MIoUMetric(Metric):
    """
    mIoU Metric
    """

    def __init__(
        self,
        class_names,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.class_names = class_names
        self.num_classes = len(class_names)

        self.reset()

    def name(self):
        """
        Return name of metric instance.
        """
        return self.__name__

    def reset(self):
        """reset"""
        self.tp_buffer = [0] * self.num_classes
        self.fp_buffer = [0] * self.num_classes
        self.fn_buffer = [0] * self.num_classes

    def update(self, results):
        """
        update

        Args:
            result (dict|list[dict]): result dict

        Return:
            None
        """
        if not isinstance(results, list):
            results = [results]

        for result in tqdm.tqdm(results):
            pred_label = result["pred_label"]
            gt_label = result["gt_label"].astype(pred_label.dtype)

            # for each class
            for class_id in range(self.num_classes):
                tp = ((gt_label == class_id) & (pred_label == class_id)).sum()
                fp = ((gt_label != class_id) & (pred_label == class_id)).sum()
                fn = ((gt_label == class_id) & (pred_label != class_id)).sum()

                self.tp_buffer[class_id] += tp.item()
                self.fp_buffer[class_id] += fp.item()
                self.fn_buffer[class_id] += fn.item()

    def accumulate(self, save_dir=None) -> dict:
        """
        accumulate

        Args:
            save_dir (str): save dir for metric curve

        Return:
            ap_dict (dict): ap dict
        """
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

        # collect results from all ranks
        buffer = dict(
            tp_buffer=self.tp_buffer,
            fp_buffer=self.fp_buffer,
            fn_buffer=self.fn_buffer,
        )
        ret_list = api_utils.collect_object(buffer)
        self.reset()
        for ret in ret_list:
            for class_id in range(self.num_classes):
                self.tp_buffer[class_id] += ret["tp_buffer"][class_id]
                self.fp_buffer[class_id] += ret["fp_buffer"][class_id]
                self.fn_buffer[class_id] += ret["fn_buffer"][class_id]

        metrics = dict()
        for class_id in range(self.num_classes):
            class_name = self.class_names[class_id]
            tp = self.tp_buffer[class_id]
            fp = self.fp_buffer[class_id]
            fn = self.fn_buffer[class_id]

            IoU = tp / (tp + fp + fn)
            metrics[class_name] = IoU

        metrics["mIoU"] = np.mean(list(metrics.values()))
        keys = ["mIoU"] + self.class_names
        values = [metrics[key] for key in keys]
        values = ["{:.2f}".format(value * 100) for value in values]
        print("| " + " | ".join(keys) + " |")
        print("|" + " :---: |" * len(keys))
        print("| " + " | ".join(values) + " |")

        return metrics
