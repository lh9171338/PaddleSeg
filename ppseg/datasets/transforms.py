# -*- encoding: utf-8 -*-
"""
@File    :   transforms.py
@Time    :   2023/11/26 15:19:21
@Author  :   lihao57
@Version :   1.0
@Contact :   lihao57@baidu.com
"""


import os
import cv2
import numpy as np
import random
import pickle
import logging
import paddle.vision.transforms as T
from ppseg.apis import manager


__all__ = [
    "Compose",
    "LoadImageFromFile",
    "LoadLabelFromFile",
    "ColorJitter",
    "RandomVerticalFlip",
    "RandomHorizontalFlip",
    "ResizeImage",
    "RandomScaleImage",
    "RandomErasingImage",
    "RandomCropImage",
    "NormalizeImage",
    "Visualize",
]


class Compose:
    """
    Compose
    """

    def __init__(self, transforms):
        if not isinstance(transforms, list):
            raise TypeError("The transforms must be a list!")
        self.transforms = transforms

    def __call__(self, sample):
        """ """
        for t in self.transforms:
            sample = t(sample)

        return sample


@manager.TRANSFORMS.add_component
class LoadImageFromFile:
    """
    Load image from file
    """

    def __init__(self, to_float=False):
        self.to_float = to_float

    def __call__(self, sample):
        image_file = sample["img_meta"]["image_file"]
        image = cv2.imread(image_file)
        if self.to_float:
            image = image.astype("float32")

        sample["image"] = image
        sample["img_meta"].update(
            dict(
                ori_size=(image.shape[1], image.shape[0]),
                img_size=(image.shape[1], image.shape[0]),
            )
        )

        return sample


@manager.TRANSFORMS.add_component
class LoadLabelFromFile:
    """
    Load label from file
    """

    def __init__(self, to_float=False):
        self.to_float = to_float

    def __call__(self, sample):
        label_file = sample["img_meta"]["label_file"]
        if not label_file:
            return sample

        label = cv2.imread(label_file, cv2.IMREAD_GRAYSCALE)
        if self.to_float:
            label = label.astype("float32")
        sample["label"] = label

        return sample


@manager.TRANSFORMS.add_component
class ColorJitter:
    """
    ColorJitter
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.transform = T.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, sample):
        sample["image"] = self.transform(sample["image"])

        return sample


@manager.TRANSFORMS.add_component
class RandomVerticalFlip:
    """
    Random vertical flip
    """

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        if random.random() < self.prob:
            sample["image"] = sample["image"][::-1]
            if "label" in sample:
                sample["label"] = sample["label"][::-1]

        return sample


@manager.TRANSFORMS.add_component
class RandomHorizontalFlip:
    """
    Random horizontal flip
    """

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        if random.random() < self.prob:
            sample["image"] = sample["image"][:, ::-1]
            if "label" in sample:
                sample["label"] = sample["label"][:, ::-1]

        return sample


@manager.TRANSFORMS.add_component
class ResizeImage:
    """
    Resize image
    """

    def __init__(self, size, interp=cv2.INTER_LINEAR):
        self.size = size
        self.interp = interp

    def __call__(self, sample):
        sample["image"] = cv2.resize(
            sample["image"], self.size, interpolation=self.interp
        )
        sample["img_meta"]["img_size"] = self.size
        if "label" in sample:
            sample["label"] = cv2.resize(
                sample["label"], self.size, interpolation=cv2.INTER_NEAREST
            )

        return sample


@manager.TRANSFORMS.add_component
class RandomScaleImage:
    """
    Random scale image
    """

    def __init__(self, scales, interp=cv2.INTER_LINEAR):
        assert len(scales) == 2, "len of scales should be 2"
        self.scales = scales
        self.interp = interp

    def __call__(self, sample):
        scale = (
            random.random() * (self.scales[1] - self.scales[0])
            + self.scales[0]
        )
        image = cv2.resize(
            sample["image"],
            (0, 0),
            fx=scale,
            fy=scale,
            interpolation=self.interp,
        )
        sample["image"] = image
        sample["img_meta"]["img_size"] = (image.shape[1], image.shape[0])
        if "label" in sample:
            sample["label"] = cv2.resize(
                sample["label"],
                (0, 0),
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_NEAREST,
            )

        return sample


@manager.TRANSFORMS.add_component
class RandomErasingImage:
    """
    Random erasing image
    """

    def __init__(
        self, prob=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0
    ):
        self.transform = T.RandomErasing(prob, scale, ratio, value)

    def __call__(self, sample):
        sample["image"] = self.transform(sample["image"])

        return sample


@manager.TRANSFORMS.add_component
class RandomCropImage:
    """
    Random crop image
    """

    def __init__(self, size, prob=0.5):
        self.size = size
        self.prob = prob

    def __call__(self, sample):
        img_size = sample["img_meta"]["img_size"]
        assert (
            img_size[0] >= self.size[0] and img_size[1] >= self.size[1]
        ), "img_size should be larger than crop size"
        if random.random() < self.prob:
            offset_x = random.randint(0, img_size[0] - self.size[0])
            offset_y = random.randint(0, img_size[1] - self.size[1])
            sample["image"] = sample["image"][
                offset_y : offset_y + self.size[1],
                offset_x : offset_x + self.size[0],
            ]
            sample["img_meta"]["img_size"] = self.size

            if "label" in sample:
                sample["label"] = sample["label"][
                    offset_y : offset_y + self.size[1],
                    offset_x : offset_x + self.size[0],
                ]

        return sample


@manager.TRANSFORMS.add_component
class NormalizeImage:
    """
    Normalize image
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image = sample["image"].astype("float32")
        image = (image - self.mean) / self.std
        sample["image"] = image.transpose((2, 0, 1)).astype("float32")

        return sample


@manager.TRANSFORMS.add_component
class Visualize:
    """
    Visualize

    label 0 is background
    """

    def __init__(
        self, save_path, class_names=None, with_label=False, alpha=0.5
    ):
        os.makedirs(save_path, exist_ok=True)
        self.save_path = save_path
        self.class_names = class_names
        self.with_label = with_label
        self.alpha = alpha

        num_classes = len(class_names) - 1
        colors = np.ones((1, num_classes, 3), dtype="float32")
        colors[:, :, 0] = np.linspace(0, 360, num_classes)
        colors = cv2.cvtColor(colors, cv2.COLOR_HSV2RGB)
        colors = (colors * 255).astype("uint8")
        colors = [color.tolist() for color in colors[0]]
        self.color_dict = dict(zip(class_names[1:], colors))

    def __call__(self, sample):
        image = sample["image"].copy()
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        if self.with_label and "label" in sample:
            label = sample["label"]
            for i in range(1, len(self.class_names)):
                mask = label == i
                color = self.color_dict[self.class_names[i]]
                image2 = np.zeros_like(image)
                image2[mask] = color
                image = cv2.addWeighted(
                    image, 1 - self.alpha, image2, self.alpha, 0
                )

        filename = os.path.basename(sample["img_meta"]["image_file"])
        save_file = os.path.join(self.save_path, filename)
        cv2.imwrite(save_file, image)

        return sample
