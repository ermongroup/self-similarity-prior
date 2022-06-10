import typing as T

import chex
import ml_collections
import jax.numpy as jnp

ArrayTree = T.Any  # chex.ArrayTree is not supported yet
MiniBatch = ArrayTree
Variables = ArrayTree
Params = ArrayTree
ModelState = ArrayTree
PRNGDict = T.Dict[str, chex.PRNGKey]
ForwardFn = T.Callable[[Variables, MiniBatch, T.Optional[PRNGDict]],
                       T.Tuple[jnp.ndarray, T.Tuple[chex.ArrayTree, ModelState]]]
ConfigDict = ml_collections.config_dict.config_dict.ConfigDict