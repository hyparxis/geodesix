import jax
import jax.numpy as jnp


def skew(vec: jax.Array) -> jax.Array:
    """Return the skew-symmetric matrix of a 3D vector."""
    if vec.shape != (3, 1):
        raise ValueError(f"Expected a vector to have shape (3, 1), got {vec.shape}")

    return jnp.array(
        [
            [0, -vec[2], vec[1]],
            [vec[2], 0, -vec[0]],
            [-vec[1], vec[0], 0],
        ]
    )
