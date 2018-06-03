#!/usr/bin/env python
"""
Gets trained model and warms it up (i.e. compiles and dumps corresponding prediction functions)
"""

import os
import sys

import botocore
import boto3

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cakechat.utils.env import init_theano_env

init_theano_env()

from cakechat.dialog_model.factory import get_trained_model
from cakechat.utils.logger import get_tools_logger
from cakechat import config

_logger = get_tools_logger(__file__)

PATHS = [["tokens_index", "t_idx_processed_dialogs.json"]]


def download_files(paths):
    bucket_client = boto3.resource(
        "s3",
        config=botocore.client.Config(signature_version=botocore.UNSIGNED),
    ).Bucket(config.S3_MODELS_BUCKET_NAME)

    for file in files:
        print("aaa")


if __name__ == "__main__":
    download_files(paths)