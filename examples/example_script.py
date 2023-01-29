import mlxu



FLAGS, FLAGS_DEF = mlxu.define_flags_with_default(
    name='example_experiment',
    seed=42,
    use_mlxu=True,
    nested_config=mlxu.config_dict(
        name='test',
        x=3,
        y=4.2
    ),
    logger=mlxu.WandBLogger.get_default_config(),
)


def main(argv):
    mlxu.print_flags(FLAGS, FLAGS_DEF)
    logger = mlxu.WandBLogger(FLAGS.logger, mlxu.get_user_flags(FLAGS, FLAGS_DEF))
    logger.log({'x': 1.0, 'y': 3.0})


if __name__ == "__main__":
    mlxu.run(main)
