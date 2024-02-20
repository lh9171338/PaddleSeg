# -*- encoding: utf-8 -*-
"""
@File    :   dove_dataset.py
@Time    :   2024/02/18 16:37:20
@Author  :   lihao57
@Version :   1.0
@Contact :   lihao57@baidu.com
"""


import os
import logging
import numpy as np
import paddle
from ppseg.datasets import BaseDataset
from ppseg.apis import manager


@manager.DATASETS.add_component
class DoveDataset(BaseDataset):
    """
    Dove Dataset
    """

    def __init__(
        self,
        data_root,
        ann_file,
        mode,
        class_names,
        pipeline=None,
        **kwargs,
    ):
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            mode=mode,
            class_names=class_names,
            pipeline=pipeline,
        )

        logging.info("len of {} dataset is: {}".format(self.mode, len(self)))

    def __getitem__(self, index):
        info = self.data_infos[index]
        image_file = os.path.join(self.data_root, info["image_file"])

        sample = {
            "mode": self.mode,
            "img_meta": {
                "sample_idx": index,
                "epoch": self.epoch,
                "image_file": image_file,
            },
        }
        if not self.is_test_mode:
            label_file = os.path.join(self.data_root, info["label_file"])
            sample["img_meta"]["label_file"] = label_file

        if self.pipeline:
            sample = self.pipeline(sample)

        return sample

    def collate_fn(self, batch):
        """collate_fn"""
        sample = batch[0]
        collated_batch = {}
        for key in sample:
            if key in ["image", "label"]:
                collated_batch[key] = np.stack(
                    [elem[key] for elem in batch], axis=0
                )
            elif key in ["img_meta"]:
                collated_batch[key] = [elem[key] for elem in batch]

        return collated_batch


if __name__ == "__main__":
    # set base logging config
    fmt = "[%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s] %(message)s"
    logging.basicConfig(format=fmt, level=logging.INFO)

    from ppseg.datasets.transforms import (
        LoadImageFromFile,
        LoadLabelFromFile,
        ColorJitter,
        RandomHorizontalFlip,
        RandomVerticalFlip,
        ResizeImage,
        RandomScaleImage,
        RandomCropImage,
        Visualize,
        NormalizeImage,
    )

    class_names = [
        "Background",
        "Dove",
    ]
    dataset = {
        "type": "DoveDataset",
        "data_root": "data/dove",
        "ann_file": "data/dove/train.pkl",
        "mode": "train",
        "class_names": class_names,
        "pipeline": [
            LoadImageFromFile(),
            LoadLabelFromFile(),
            # CopyAndPalse(
            #     class_names=class_names,
            #     db_sampler_file="data/dove/dbsampler.pkl",
            #     num_samples=5,
            # ),
            # ColorJitter(0.4, 0.4, 0.4, 0.4),
            RandomHorizontalFlip(0.5),
            RandomVerticalFlip(0.5),
            # RandomCropImage((640, 640), 0.5),
            # ResizeImage(size=(640, 640)),
            # RandomScaleImage([0.5, 2]),
            Visualize(
                save_path="visualize", class_names=class_names, with_label=True
            ),
            NormalizeImage(
                mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]
            ),
        ],
    }
    dataset = DoveDataset(**dataset)
    dataloader = paddle.io.DataLoader(
        dataset,
        batch_size=4,
        num_workers=16,
        shuffle=True,
        collate_fn=dataset.collate_fn,
    )
    for sample in dataset:
        print(sample["img_meta"]["sample_idx"])
