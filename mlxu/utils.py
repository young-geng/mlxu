import os
import random
import tempfile
import time
import uuid
from copy import copy
from socket import gethostname
import logging
from io import BytesIO

import numpy as np
import cloudpickle as pickle
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


def open_file(path, mode='rb', cache_type='readahead'):
    if path.startswith("gs://"):
        logging.getLogger("fsspec").setLevel(logging.WARNING)
        return gcsfs.GCSFileSystem().open(path, mode, cache_type=cache_type)
    else:
        return open(path, mode)


def save_pickle(obj, path):
    with open_file(path, 'wb') as fout:
        pickle.dump(obj, fout)


def load_pickle(path):
    with open_file(path, 'rb') as fin:
        data = pickle.load(fin)
    return data


def text_to_array(text, encoding='utf-8'):
    return np.frombuffer(text.encode(encoding), dtype='uint8')


def array_to_text(array, encoding='utf-8'):
    with BytesIO(array) as fin:
        text = fin.read().decode(encoding)
    return text
