import os
import pprint
import random
import tempfile
import time
import inspect

import absl.flags
from absl import logging
from ml_collections import ConfigDict
from ml_collections.config_dict.config_dict import placeholder
from ml_collections.config_flags import config_flags

def config_dict(*args, **kwargs):
    return ConfigDict(dict(*args, **kwargs))


def define_flags_with_default(**kwargs):
    for key, val in kwargs.items():
        if isinstance(val, ConfigDict):
            config_flags.DEFINE_config_dict(key, val)
        elif isinstance(val, bool):
            # Note that True and False are instances of int.
            absl.flags.DEFINE_bool(key, val, "automatically defined flag")
        elif isinstance(val, int):
            absl.flags.DEFINE_integer(key, val, "automatically defined flag")
        elif isinstance(val, float):
            absl.flags.DEFINE_float(key, val, "automatically defined flag")
        elif isinstance(val, str):
            absl.flags.DEFINE_string(key, val, "automatically defined flag")
        else:
            raise ValueError("Incorrect value type")
    return absl.flags.FLAGS, kwargs


def print_flags(flags, flags_def):
    flag_srings = [
        "{}: {}".format(key, val)
        for key, val in get_user_flags(flags, flags_def).items()
    ]
    logging.info(
        "Hyperparameter configs: \n{}".format(
            pprint.pformat(flag_srings)
        )
    )


def get_user_flags(flags, flags_def):
    output = {}
    for key in flags_def:
        val = getattr(flags, key)
        if isinstance(val, ConfigDict):
            output.update(flatten_config_dict(val, prefix=key))
        else:
            output[key] = val

    return output


def user_flags_to_config_dict(flags, flags_def):
    output = ConfigDict()
    for key in flags_def:
        output[key] = getattr(flags, key)

    return output


def flatten_config_dict(config, prefix=None):
    output = {}
    for key, val in config.items():
        if isinstance(val, ConfigDict) or isinstance(val, dict):
            output.update(flatten_config_dict(val, prefix=key))
        else:
            if prefix is not None:
                output["{}.{}".format(prefix, key)] = val
            else:
                output[key] = val
    return output


def function_args_to_config(fn, none_arg_types=None, exclude_args=None, override_args=None):
    config = ConfigDict()
    arg_spec = inspect.getargspec(fn)
    n_args = len(arg_spec.defaults)
    arg_names = arg_spec.args[-n_args:]
    default_values = arg_spec.defaults
    for name, value in zip(arg_names, default_values):
        if exclude_args is not None and name in exclude_args:
            continue
        elif override_args is not None and name in override_args:
            config[name] = override_args[name]
        elif none_arg_types is not None and value is None and name in none_arg_types:
            config[name] = placeholder(none_arg_types[name])
        else:
            config[name] = value

    return config
