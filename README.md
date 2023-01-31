# MLXU: Machine Learning eXperiment Utilities
This library provide a collection of utilities for machine learning experiments.
MLXU is a thin wrapper on top of [absl-py](https://github.com/abseil/abseil-py),
[ml_collections](https://github.com/google/ml_collections) and
[wandb](https://github.com/wandb/wandb). It also provides some convenient JAX
utils.


This library includes the following modules:
 * [config](mlxu/config.py) Experiment configuration and command line flags utils
 * [logging](mlxu/logging.py) W&B logging utils
 * [jax_utils](mlxu/jax_utils.py) JAX specific utils


# Installation
MLXU can be installed via pip. To install from PYPI
```shell
pip install mlxu
```

To install the latest version from GitHub
```shell
pip install git+https://github.com/young-geng/mlxu.git
```


# Examples
Here are some examples for the utilities provide in MLXU

## Command Line Flags and Logging
```python
import mlxu

# Define absl command line flags in one function, with automatic type inference.
FLAGS, FLAGS_DEF = mlxu.define_flags_with_default(
    name='example_experiment',          # string flag
    seed=42,                            # integer flag
    learning_rate=1e-3,                 # floating point flag
    use_mlxu=True,                      # boolean flag
    network_architecture=mlxu.config_dict(
        activation='relu',
        hidden_dim=128,
        hidden_layers=5,
    ),                                  # nest ml_collections config_dict
    logger=mlxu.WandBLogger.get_default_config(),  # logger configuration
)


def main(argv):
    # Print the command line flags
    mlxu.print_flags(FLAGS, FLAGS_DEF)

    # Access the flags
    name = FLAGS.name
    seed = FLAGS.seed

    # Access nested flags
    activation = FLAGS.network_architecture.activation
    hidden_dim = FLAGS.network_architecture.hidden_dim

    # Create logger and log metrics
    logger = mlxu.WandBLogger(FLAGS.logger, mlxu.get_user_flags(FLAGS, FLAGS_DEF))
    logger.log({'step': 1, 'loss': 10.5})


# Run the main function
if __name__ == "__main__":
    mlxu.run(main)
```
