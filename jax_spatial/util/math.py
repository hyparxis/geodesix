import jax
import jax.numpy as jnp


def skew(vec: jax.Array) -> jax.Array:
    if vec.shape != (3, 1):
        raise ValueError(f"Expected shape (3, 1), got {vec.shape}")

    x = vec[0, 0]
    y = vec[1, 0]
    z = vec[2, 0]

    return jnp.array(
        [
            [0, -z, y],
            [z, 0, -x],
            [-y, x, 0],
        ]
    )


def deskew(m: jax.Array) -> jax.Array:
    """
    Extract the (x, y, z) vector from a 3x3 skew-symmetric matrix
    (inverse of skew)
    """
    if m.shape != (3, 3):
        raise ValueError(f"Expected shape (3, 3), got {m.shape}")

    tau = jnp.array(
        [
            m[2, 1] - m[1, 2],
            m[0, 2] - m[2, 0],
            m[1, 0] - m[0, 1],
        ]
    )
    return 0.5 * tau
