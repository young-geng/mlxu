import dataclasses
import random

import numpy as np
import jax
import jax.numpy as jnp


class JaxRNG(object):
    """ A convenient stateful Jax RNG wrapper. Can be used to wrap RNG inside
        pure function.
    """
    global_rng_generator = None

    @classmethod
    def from_seed(cls, seed):
        return cls(jax.random.PRNGKey(seed))

    def __init__(self, rng):
        self.rng = rng

    def __call__(self, keys=None):
        if keys is None:
            self.rng, split_rng = jax.random.split(self.rng)
            return split_rng
        elif isinstance(keys, int):
            split_rngs = jax.random.split(self.rng, num=keys + 1)
            self.rng = split_rngs[0]
            return tuple(split_rngs[1:])
        else:
            split_rngs = jax.random.split(self.rng, num=len(keys) + 1)
            self.rng = split_rngs[0]
            return {key: val for key, val in zip(keys, split_rngs[1:])}

    @classmethod
    def init_global_rng(cls, seed):
        cls.global_rng_generator = cls.from_seed(seed)

    @classmethod
    def next_rng(cls, *args, **kwargs):
        assert cls.global_rng_generator is not None, 'Global RNG not initialized.'
        return cls.global_rng_generator(*args, **kwargs)


def init_rng(seed):
    JaxRNG.init_global_rng(seed)


def next_rng(*args, **kwargs):
    return JaxRNG.next_rng(*args, **kwargs)


def wrap_function_with_rng(rng):
    """ To be used as decorator, automatically bookkeep a RNG for the wrapped function. """
    def wrap_function(function):
        def wrapped(*args, **kwargs):
            nonlocal rng
            rng, split_rng = jax.random.split(rng)
            return function(split_rng, *args, **kwargs)
        return wrapped
    return wrap_function


def tree_path_to_string(path, sep=None):
    keys = []
    for key in path:
        if isinstance(key, jax.tree_util.SequenceKey):
            keys.append(str(key.idx))
        elif isinstance(key, jax.tree_util.DictKey):
            keys.append(str(key.key))
        elif isinstance(key, jax.tree_util.GetAttrKey):
            keys.append(str(key.name))
        elif isinstance(key, jax.tree_util.FlattenedIndexKey):
            keys.append(str(key.key))
        else:
            keys.append(str(key))
    if sep is None:
        return tuple(keys)
    return sep.join(keys)


def flatten_tree(xs, is_leaf=None, sep=None):
    flattened, _ = jax.tree_util.tree_flatten_with_path(xs, is_leaf=is_leaf)
    output = {}
    for key, val in flattened:
        output[tree_path_to_string(key, sep=sep)] = val
    return output


def named_tree_map(f, tree, *rest, is_leaf=None, sep=None):
    """ An extended version of jax.tree_util.tree_map, where the mapped function
        f takes both the name (path) and the tree leaf as input.
    """
    return jax.tree_util.tree_map_with_path(
        lambda path, x, *r: f(tree_path_to_string(path, sep=sep), x, *r),
        tree, *rest,
        is_leaf=is_leaf
    )


def get_pytree_shape_info(tree):
    flattend_tree = flatten_tree(tree, sep='/')
    shapes = []
    for key in sorted(list(flattend_tree.keys())):
        val = flattend_tree[key]
        shapes.append(f'{key}: {val.dtype}, {val.shape}')
    return '\n'.join(shapes)


def collect_metrics(metrics, names, prefix=None):
    collected = {}
    for name in names:
        if name in metrics:
            collected[name] = jnp.mean(metrics[name])
    if prefix is not None:
        collected = {
            '{}/{}'.format(prefix, key): value for key, value in collected.items()
        }
    return collected


def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    init_rng(seed)
