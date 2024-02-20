# -*- encoding: utf-8 -*-
"""
@File    :   visualizer.py
@Time    :   2023/12/21 22:46:13
@Author  :   lihao57
@Version :   1.0
@Contact :   lihao57@baidu.com
"""


import os
import cv2
import numpy as np
from ppseg.apis import manager
import ppseg.apis.utils as api_utils
from lh_tool.Iterator import SingleProcess, MultiProcess


__all__ = ["Visualizer", "SegVisualizer"]


class Visualizer:
    """
    Visualizer

    label 0 is background
    """

    def __init__(
        self,
        class_names,
        alpha=0.5,
        nprocs=1,
        **kwargs,
    ):
        self.class_names = class_names
        self.alpha = alpha
        self.nprocs = nprocs

        num_classes = len(class_names) - 1
        colors = np.ones((1, num_classes, 3), dtype="float32")
        colors[:, :, 0] = np.linspace(0, 360, num_classes)
        colors = cv2.cvtColor(colors, cv2.COLOR_HSV2RGB)
        colors = (colors * 255).astype("uint8")
        colors = [color.tolist() for color in colors[0]]
        self.color_dict = dict(zip(class_names[1:], colors))

        self.reset()

    def reset(self):
        """reset"""
        self.result_buffer = []

    def update(self, results):
        """
        update

        Args:
            results (dict|list[dict]): prediction and target
        """
        if not isinstance(results, list):
            results = [results]
        results = api_utils.tensor2numpy(results)
        self.result_buffer.extend(results)

    def visualize(self, save_dir):
        """
        visualize

        Args:
            save_dir (str): save directory

        Returns:
            None
        """
        raise NotImplementedError


@manager.VISUALIZERS.add_component
class SegVisualizer(Visualizer):
    """
    Segmentation Visualizer
    """

    def _visualize_single(self, save_dir, result):
        """
        visualize single image

        Args:
            save_dir (str): save directory
            result (dict): prediction and target

        Returns:
            None
        """
        # read image
        img_size = result["img_size"]
        image_file = result["image_file"]
        image = cv2.imread(image_file)
        image = cv2.resize(image, img_size)

        # pred
        if "pred_label" in result:
            label = result["pred_label"]
            for i in range(1, len(self.class_names)):
                mask = label == i
                color = self.color_dict[self.class_names[i]]
                image2 = np.zeros_like(image)
                image2[mask] = color
                image = cv2.addWeighted(
                    image, 1 - self.alpha, image2, self.alpha, 0
                )

        # save
        save_file = os.path.join(save_dir, os.path.basename(image_file))
        cv2.imwrite(save_file, image)

    def visualize(self, save_dir):
        """
        visualize

        Args:
            save_dir (str): save directory

        Returns:
            None
        """
        os.makedirs(save_dir, exist_ok=True)
        process = MultiProcess if self.nprocs > 1 else SingleProcess
        process(self._visualize_single, nprocs=self.nprocs).run(
            save_dir=save_dir,
            result=self.result_buffer,
        )
