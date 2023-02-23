import mlxu

# Define absl command line flags in one function, with automatic type inference.
FLAGS, FLAGS_DEF = mlxu.define_flags_with_default(
    name='example_experiment',          # string flag
    seed=42,                            # integer flag
    learning_rate=1e-3,                 # floating point flag
    use_mlxu=True,                      # boolean flag
    epochs=(10, 'Number of epochs'),    # we can also specify help strings
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