# -*- encoding: utf-8 -*-
"""
@File    :   metric.py
@Time    :   2023/12/21 22:32:32
@Author  :   lihao57
@Version :   1.0
@Contact :   lihao57@baidu.com
"""


from paddle.metric import Metric
from ppseg.apis import manager


__all__ = [
    "ComposeMetric",
    "MockMetric",
]


@manager.METRICS.add_component
class ComposeMetric(Metric):
    """
    Compose Metric
    """

    def __init__(
        self,
        metrics,
        **kwargs,
    ):
        super().__init__(**kwargs)

        assert len(metrics) > 0, "metric_dict should not be empty."
        self.metrics = metrics

        self.reset()

    def name(self):
        """
        Return name of metric instance.
        """
        return self.__name__

    def reset(self):
        """reset"""
        for metric in self.metrics:
            metric.reset()

    def update(self, results):
        """
        update

        Args:
            result (dict|list[dict]): result dict

        Return:
            None
        """
        for metric in self.metrics:
            metric.update(results)

    def accumulate(self, save_dir):
        """
        accumulate

        Args:
            save_dir (str): save dir for metric curve

        Return:
            metric_dict (dict): metric dict
        """
        metric_dict = dict()
        for metric in self.metrics:
            metric_dict.update(metric.accumulate(save_dir))

        return metric_dict


@manager.METRICS.add_component
class MockMetric(Metric):
    """
    Mock Metric
    """

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

    def name(self):
        """
        Return name of metric instance.
        """
        return self.__name__

    def reset(self):
        """reset"""
        pass

    def update(self, results):
        """
        update

        Args:
            result (dict|list[dict]): result dict

        Return:
            None
        """
        pass

    def accumulate(self, save_dir):
        """
        accumulate

        Args:
            save_dir (str): save dir for metric curve

        Return:
            metric_dict (dict): metric dict
        """
        metric_dict = dict()

        return metric_dict
