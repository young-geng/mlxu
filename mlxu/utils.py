import os
import random
import tempfile
import time
import uuid
from copy import copy
from socket import gethostname

import cloudpickle as pickle
import gcsfs
import wandb
from absl import logging
from ml_collections import ConfigDict
from ml_collections.config_dict import config_dict
import gcsfs


class Timer(object):
    def __init__(self):
        self._time = None

    def __enter__(self):
        self._start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self._time = time.time() - self._start_time

    def __call__(self):
        return self._time


def open_file(path, mode='rb'):
    if path.startswith("gs://"):
        return gcsfs.GCSFileSystem().open(path, mode, cache_type='block')
    else:
        return open(path, mode)
