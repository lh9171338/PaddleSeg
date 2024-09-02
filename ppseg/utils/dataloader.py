# -*- encoding: utf-8 -*-
"""
@File    :   dataloader.py
@Time    :   2024/08/28 21:25:46
@Author  :   lihao57
@Version :   1.0
@Contact :   lihao57@baidu.com
"""


import paddle


class MultiEpochsDataLoader(paddle.io.DataLoader):
    """MultiEpochsDataLoader"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)
