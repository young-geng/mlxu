import mlxu


class ConfigurableModule(object):
    # Define a configurable module with a default configuration. This module
    # can be directly configured from the command line when plugged into
    # the FLAGS.

    @staticmethod
    def get_default_config(updates=None):
        config = mlxu.config_dict()
        config.integer_value = 10
        config.float_value = 1.0
        config.string_value = 'hello'
        config.boolean_value = True
        return mlxu.update_config_dict(config, updates)

    def __init__(self, config):
        self.config = self.get_default_config(config)


# Define absl command line flags in one function, with automatic type inference.
FLAGS, FLAGS_DEF = mlxu.define_flags_with_default(
    name='example_experiment',          # string flag
    seed=42,                            # integer flag
    learning_rate=1e-3,                 # floating point flag
    use_mlxu=True,                      # boolean flag
    num_gpus=int,                       # integer flag without default value
    weight_decay=float,                 # floating point flag without default value
    save_checkpoints=bool,              # boolean flag without default value
    epochs=(10, 'Number of epochs'),    # we can also specify help strings
    network_architecture=mlxu.config_dict(
        activation='relu',
        hidden_dim=128,
        hidden_layers=5,
    ),                                  # nested ml_collections config_dict
    configurable_module=ConfigurableModule.get_default_config(),  # nested custom config_dict
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

    configurable_module = ConfigurableModule(FLAGS.configurable_module)

    # Create logger and log metrics
    logger = mlxu.WandBLogger(FLAGS.logger, mlxu.get_user_flags(FLAGS, FLAGS_DEF))
    logger.log({'step': 1, 'loss': 10.5})
    logger.save_pickle([1, 2, 4, 5], 'checkpoint.pkl')


# Run the main function
if __name__ == "__main__":
    mlxu.run(main)