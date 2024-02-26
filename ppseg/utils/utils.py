# -*- encoding: utf-8 -*-
"""
@File    :   utils.py
@Time    :   2024/02/22 14:50:53
@Author  :   lihao57
@Version :   1.0
@Contact :   lihao57@baidu.com
"""


import os
import contextlib
import tempfile
from urllib.parse import urlparse, unquote
import paddle
import logging
from ppseg.utils import seg_env
from ppseg.utils.download import download_file_and_uncompress


@contextlib.contextmanager
def generate_tempdir(directory: str = None, **kwargs):
    """Generate a temporary directory"""
    directory = seg_env.TMP_HOME if not directory else directory
    with tempfile.TemporaryDirectory(dir=directory, **kwargs) as _dir:
        yield _dir


def load_entire_model(model, pretrained):
    """
    load pretrained model

    Args:
        model (paddle.nn.Layer): model
        pretrained (str): pretrained model path or url

    Returns:
        None
    """
    if pretrained is not None:
        load_pretrained_model(model, pretrained)
    else:
        logging.warning(
            "Weights are not loaded for {} model since the "
            "path of weights is None".format(model.__class__.__name__)
        )


def download_pretrained_model(pretrained_model):
    """
    Download pretrained model from url.
    Args:
        pretrained_model (str): the url of pretrained weight
    Returns:
        str: the path of pretrained weight
    """
    assert urlparse(pretrained_model).netloc, "The url is not valid."

    pretrained_model = unquote(pretrained_model)
    savename = pretrained_model.split("/")[-1]
    if not savename.endswith(("tgz", "tar.gz", "tar", "zip")):
        savename = pretrained_model.split("/")[-2]
        filename = pretrained_model.split("/")[-1]
    else:
        savename = savename.split(".")[0]
        filename = "model.pdparams"

    with generate_tempdir() as _dir:
        pretrained_model = download_file_and_uncompress(
            pretrained_model,
            savepath=_dir,
            cover=False,
            extrapath=seg_env.PRETRAINED_MODEL_HOME,
            extraname=savename,
            filename=filename,
        )
        pretrained_model = os.path.join(pretrained_model, filename)
    return pretrained_model


def load_pretrained_model(model, pretrained_model):
    """
    load pretrained model

    Args:
        model (paddle.nn.Layer): model
        pretrained_model (str): pretrained model path or url

    Returns:
        None
    """
    if pretrained_model is not None:
        logging.info(
            "Loading pretrained model from {}".format(pretrained_model)
        )

        if urlparse(pretrained_model).netloc:
            pretrained_model = download_pretrained_model(pretrained_model)

        if os.path.exists(pretrained_model):
            para_state_dict = paddle.load(pretrained_model)

            model_state_dict = model.state_dict()
            keys = model_state_dict.keys()
            num_params_loaded = 0
            for k in keys:
                if k not in para_state_dict:
                    logging.warning("{} is not in pretrained model".format(k))
                elif list(para_state_dict[k].shape) != list(
                    model_state_dict[k].shape
                ):
                    logging.warning(
                        "[SKIP] Shape of pretrained params {} doesn't match.(Pretrained: {}, Actual: {})".format(
                            k,
                            para_state_dict[k].shape,
                            model_state_dict[k].shape,
                        )
                    )
                else:
                    model_state_dict[k] = para_state_dict[k]
                    num_params_loaded += 1
            model.set_dict(model_state_dict)
            logging.info(
                "There are {}/{} variables loaded into {}.".format(
                    num_params_loaded,
                    len(model_state_dict),
                    model.__class__.__name__,
                )
            )

        else:
            raise ValueError(
                "The pretrained model directory is not Found: {}".format(
                    pretrained_model
                )
            )
    else:
        logging.info(
            "No pretrained model to load, {} will be trained from scratch.".format(
                model.__class__.__name__
            )
        )
