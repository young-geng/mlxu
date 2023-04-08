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

from .utils import open_file, save_pickle, load_pickle
from .config import flatten_config_dict


class WandBLogger(object):
    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.online = False
        config.prefix = ""
        config.project = "mlxu"
        config.output_dir = "/tmp/mlxu"
        config.wandb_dir = ""
        config.random_delay = 0.0
        config.experiment_id = config_dict.placeholder(str)
        config.anonymous = config_dict.placeholder(str)
        config.notes = config_dict.placeholder(str)
        config.entity = config_dict.placeholder(str)
        config.prefix_to_id = False

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, variant, enable=True):
        self.enable = enable
        self.config = self.get_default_config(config)

        if self.config.experiment_id is None:
            self.config.experiment_id = uuid.uuid4().hex

        if self.config.prefix != "":
            if self.config.prefix_to_id:
                self.config.experiment_id = "{}--{}".format(
                    self.config.prefix, self.config.experiment_id
                )
            else:
                self.config.project = "{}--{}".format(self.config.prefix, self.config.project)

        if self.enable:
            if self.config.output_dir == "":
                self.config.output_dir = tempfile.mkdtemp()
            else:
                self.config.output_dir = os.path.join(
                    self.config.output_dir, self.config.experiment_id
                )
                if not self.config.output_dir.startswith("gs://"):
                    os.makedirs(self.config.output_dir, exist_ok=True)

            if self.config.wandb_dir == "":
                if not self.config.output_dir.startswith("gs://"):
                    # Use the same directory as output_dir if it is not a GCS path.
                    self.config.wandb_dir = self.config.output_dir
                else:
                    # Otherwise, use a temporary directory.
                    self.config.wandb_dir = tempfile.mkdtemp()
            else:
                # Join the wandb_dir with the experiment_id.
                self.config.wandb_dir = os.path.join(
                    self.config.wandb_dir, self.config.experiment_id
                )
                os.makedirs(self.config.wandb_dir, exist_ok=True)

        self._variant = flatten_config_dict(variant)

        if "hostname" not in self._variant:
            self._variant["hostname"] = gethostname()

        if self.config.random_delay > 0:
            time.sleep(random.uniform(0, self.config.random_delay))

        if self.enable:
            self.run = wandb.init(
                reinit=True,
                config=self._variant,
                project=self.config.project,
                dir=self.config.wandb_dir,
                id=self.config.experiment_id,
                anonymous=self.config.anonymous,
                notes=self.config.notes,
                entity=self.config.entity,
                settings=wandb.Settings(
                    start_method="thread",
                    _disable_stats=True,
                ),
                mode="online" if self.config.online else "offline",
            )
        else:
            self.run = None

    def log(self, *args, **kwargs):
        if self.enable:
            self.run.log(*args, **kwargs)

    def save_pickle(self, obj, filename):
        if self.enable:
            save_pickle(obj, os.path.join(self.config.output_dir, filename))

    @property
    def experiment_id(self):
        return self.config.experiment_id

    @property
    def variant(self):
        return self.config.variant

    @property
    def output_dir(self):
        return self.config.output_dir

    @property
    def wandb_dir(self):
        return self.config.wandb_dir


def prefix_metrics(metrics, prefix):
    return {"{}/{}".format(prefix, key): value for key, value in metrics.items()}
