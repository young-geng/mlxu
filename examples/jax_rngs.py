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
    jax_utils.set_random_seed(42)

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