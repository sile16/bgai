import numpy as np
import jax.numpy as jnp

from bgai.endgame.lookup import BearoffTable


def test_lookup_jax_handles_instance_static_arg():
    """lookup_jax should be callable without JAX complaining about self."""
    # Bypass __init__ to avoid allocating the full production table
    bo = object.__new__(BearoffTable)
    bo.n = 1
    bo.table = np.array([[1.0]], dtype=np.float32)
    bo._table_jax = jnp.array(bo.table)
    bo.format = "full"

    x_pos = jnp.array([0, 0, 0, 0, 0, 0], dtype=jnp.int32)
    o_pos = jnp.array([0, 0, 0, 0, 0, 0], dtype=jnp.int32)

    # Should return scalar 1.0 and not raise static-arg errors
    val = bo.lookup_jax(x_pos, o_pos)
    assert float(val) == 1.0
