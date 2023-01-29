import mlxu



FLAGS_DEF, FLAGS = mlxu.define_flags_with_default(
    name='example_experiment',
    seed=42,
    use_mlxu=True,
    nested_config=mlxu.config_dict(
        name='test',
        x=3,
        y=4.2
    )
)


def main(argv):
    mlxu.print_flags(FLAGS, FLAGS_DEF)


if __name__ == "__main__":
    mlxu.run(main)
