# -*- encoding: utf-8 -*-
"""
@File    :   seg_env.py
@Time    :   2024/02/22 14:57:16
@Author  :   lihao57
@Version :   1.0
@Contact :   lihao57@baidu.com
"""


"""
This module is used to store environmental parameters in PaddleSeg.

SEG_HOME : Root directory for storing PaddleSeg related data. Default to ~/.paddleseg.
           Users can change the default value through the SEG_HOME environment variable.
DATA_HOME : The directory to store the automatically downloaded dataset, e.g ADE20K.
PRETRAINED_MODEL_HOME : The directory to store the automatically downloaded pretrained model.
"""

import os


def _get_user_home():
    return os.path.expanduser("~")


def _get_seg_home():
    if "SEG_HOME" in os.environ:
        home_path = os.environ["SEG_HOME"]
        if os.path.exists(home_path):
            if os.path.isdir(home_path):
                return home_path
            else:
                logger.warning("SEG_HOME {} is a file!".format(home_path))
        else:
            return home_path
    return os.path.join(_get_user_home(), ".paddleseg")


def _get_sub_home(directory):
    home = os.path.join(_get_seg_home(), directory)
    if not os.path.exists(home):
        os.makedirs(home, exist_ok=True)
    return home


USER_HOME = _get_user_home()
SEG_HOME = _get_seg_home()
DATA_HOME = _get_sub_home("dataset")
TMP_HOME = _get_sub_home("tmp")
PRETRAINED_MODEL_HOME = _get_sub_home("pretrained_model")
