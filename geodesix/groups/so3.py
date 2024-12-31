import jax
import jax.numpy as jnp

from ..util.math import deskew, skew
from . import s3


def log(rotation: jax.Array) -> jax.Array:
    q = s3.Quaternion.from_rotation_matrix(rotation)
    return s3.log(q)


def hat(tau: jax.Array) -> jax.Array:
    return skew(tau)


def vee(m: jax.Array) -> jax.Array:
    return deskew(m)


def left_jacobian(tau: jax.Array) -> jax.Array:
    """
    Compute the left Jacobian of the SO(3) exponential map at tau.
    I + (1 - cos(theta))/theta^2 * W + (theta - sin(theta))/(theta^3) * W^2
    """

    eps = 1e-14
    theta_sq = jnp.sum(tau * tau)
    W = hat(tau)

    def small_angle():
        # ~ I + 1/2 * W
        return jnp.eye(3) + 0.5 * W

    def general_angle(_):
        theta = jnp.sqrt(theta_sq)
        one_minus_cos = 1.0 - jnp.cos(theta)
        sin_theta = jnp.sin(theta)
        A = one_minus_cos / theta_sq
        B = (theta - sin_theta) / (theta_sq * theta)
        return jnp.eye(3) + A * W + B * (W @ W)

    return jax.lax.cond(theta_sq <= eps, small_angle, general_angle, operand=None)


def left_jacobian_inverse(tau: jax.Array) -> jax.Array:
    eps = 1e-14
    theta_sq = jnp.sum(tau * tau)
    W = hat(tau)

    def small_angle(_):
        return jnp.eye(3) - 0.5 * W

    def general_angle(_):
        theta = jnp.sqrt(theta_sq)
        cos_theta = jnp.cos(theta)
        sin_theta = jnp.sin(theta)
        term = 1.0 / theta_sq - (1.0 + cos_theta) / (2.0 * theta * sin_theta)
        return jnp.eye(3) - 0.5 * W + term * (W @ W)

    return jax.lax.cond(theta_sq <= eps, small_angle, general_angle, operand=None)


@jax.tree_util.register_pytree_node_class
class SO3:
    """
    A class representing an SO(3) rotation matrix.
    """

    def __init__(self, rotation: jax.Array, from_frame: str, to_frame: str):
        if rotation.shape != (3, 3):
            raise ValueError(
                f"Expected 3x3 rotation matrix, got shape {rotation.shape}"
            )

        self.from_frame = from_frame
        self.to_frame = to_frame

    def log(self) -> jax.Array:
        return log(self.rotation)

    def tree_flatten(self):
        return (self.rotation,), (self.from_frame, self.to_frame)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (from_frame, to_frame) = aux_data
        (rotation,) = children
        return cls(
            rotation=rotation,
            from_frame=from_frame,
            to_frame=to_frame,
        )
