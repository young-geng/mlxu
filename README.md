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
MLXU provides convenient wrappers around absl-py and wandb to make command line
arg parsing and logging easy.
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
    logger.save_pickle([1, 2, 4, 5], 'checkpoint.pkl')


# Run the main function
if __name__ == "__main__":
    mlxu.run(main)
```

The flags can be passed in via command line arguments:
```shell
python examples/cli_logging.py \
    --name='example' \
    --seed=24 \
    --learning_rate=1.0 \
    --use_mlxu=True \
    --network_architecture.activation='gelu' \
    --network_architecture.hidden_dim=126 \
    --network_architecture.hidden_layers=2 \
    --logger.online=True \
    --logger.project='mlxu_example'
```

Specifically, the `logger.online` option controls whether the logger will upload
the data to W&B, and the `logger.project` option controls the name of the W&B
project.

## JAX Random Number Generator
MLXU also provides convenient wrapper around JAX's random number generators
to make it much easier to use
```python
import jax
import jax.numpy as jnp
import mlxu
import mlxu.jax_utils as jax_utils


@jax.jit
def sum_of_random_uniform(rng_key):
    # Capture RNG key to create a stateful rng key generator.
    # As long as JaxRNG object is not pass through the function
    # boundary, the function is still pure and jittable.
    # JaxRNG object also supports the same tuple and dictionary usage like
    # the jax_utils.next_rng function.
    rng_generator = jax_utils.JaxRNG(rng_key)
    output = jnp.zeros((2, 2))
    for i in range(4):
        # Each call returns a new key, altering the internal state of rng_generator
        output += jax.random.uniform(rng_generator(), (2, 2))

    return output


def main(argv):
    # Setup global rng generator
    jax_utils.init_rng(42)

    # Get an rng key
    rng_key = jax_utils.next_rng()
    print(rng_key)

    # Get a new rng key, this key should be different from the previous one
    rng_key = jax_utils.next_rng()
    print(rng_key)

    # You can also get a tuple of N rng keys
    k1, k2, k3 = jax_utils.next_rng(3)
    print(k1, ', ', k2, ', ', k3)

    # Dictionary of keys is also supported
    rng_key_dict = jax_utils.next_rng(['k1', 'k2'])
    print(rng_key_dict)

    # Call a jitted function that makes use of stateful JaxRNG object internally
    x = sum_of_random_uniform(jax_utils.next_rng())
    print(x)


if __name__ == "__main__":
    mlxu.run(main)
```