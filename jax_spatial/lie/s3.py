import jax
from jax import numpy as jnp


def wrap_to_pi(angle: jnp.ndarray) -> jnp.ndarray:
    """
    Wrap an angle (in radians) to the interval [-pi, pi].
    """
    # A common "mod" trick:
    return (angle + jnp.pi) % (2.0 * jnp.pi) - jnp.pi


@jax.tree_util.register_pytree_node_class
class Quaternion:
    def __init__(self, w: float, x: float, y: float, z: float):
        self.w = w
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self) -> str:
        return f"Quaternion(w={self.w}, x={self.x}, y={self.y}, z={self.z})"

    def normalized(self) -> "Quaternion":
        arr = self.as_array()
        norm = jnp.linalg.norm(arr)
        arr_norm = arr / norm
        return Quaternion(arr_norm[0], arr_norm[1], arr_norm[2], arr_norm[3])

    def tree_flatten(self):
        return (self.w, self.x, self.y, self.z), ()

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    def as_array_wxyz(self) -> jax.Array:
        return jnp.array([self.w, self.x, self.y, self.z])

    def vec(self) -> jax.Array:
        return jnp.array([self.x, self.y, self.z])

    @classmethod
    def from_rotation_matrix(cls, R: jnp.ndarray) -> "Quaternion":
        """
        Construct a quaternion from a 3x3 rotation matrix.
        """
        if R.shape != (3, 3):
            raise ValueError(f"Expected 3x3 rotation matrix, got shape {R.shape}")

        trace = R[0, 0] + R[1, 1] + R[2, 2]

        def case0():
            w = jnp.sqrt(jnp.maximum(trace + 1.0, 0.0)) / 2.0
            denom = 4.0 * w
            x = (R[2, 1] - R[1, 2]) / denom
            y = (R[0, 2] - R[2, 0]) / denom
            z = (R[1, 0] - R[0, 1]) / denom
            return Quaternion(w, x, y, z)

        def case1():
            x = jnp.sqrt(jnp.maximum((R[0, 0] - R[1, 1] - R[2, 2] + 1.0), 0.0)) / 2.0
            denom = 4.0 * x
            w = (R[2, 1] - R[1, 2]) / denom
            y = (R[0, 1] + R[1, 0]) / denom
            z = (R[0, 2] + R[2, 0]) / denom
            return Quaternion(w, x, y, z)

        def case2():
            y = jnp.sqrt(jnp.maximum((-R[0, 0] + R[1, 1] - R[2, 2] + 1.0), 0.0)) / 2.0
            denom = 4.0 * y
            w = (R[0, 2] - R[2, 0]) / denom
            x = (R[0, 1] + R[1, 0]) / denom
            z = (R[1, 2] + R[2, 1]) / denom
            return Quaternion(w, x, y, z)

        def case3():
            z = jnp.sqrt(jnp.maximum((-R[0, 0] - R[1, 1] + R[2, 2] + 1.0), 0.0)) / 2.0
            denom = 4.0 * z
            w = (R[1, 0] - R[0, 1]) / denom
            x = (R[0, 2] + R[2, 0]) / denom
            y = (R[1, 2] + R[2, 1]) / denom
            return Quaternion(w, x, y, z)

        # pick the largest among w^2, x^2, y^2, z^2 to reduce numeric issues
        w_candidate = (trace + 1.0) / 4.0
        x_candidate = (R[0, 0] - R[1, 1] - R[2, 2] + 1.0) / 4.0
        y_candidate = (-R[0, 0] + R[1, 1] - R[2, 2] + 1.0) / 4.0
        z_candidate = (-R[0, 0] - R[1, 1] + R[2, 2] + 1.0) / 4.0

        idx = jnp.argmax(
            jnp.array([w_candidate, x_candidate, y_candidate, z_candidate])
        )

        return jax.lax.switch(idx, [case0, case1, case2, case3])


def log(q: Quaternion):
    """
    Compute the 'log' map of a (unit) quaternion to R^3, mirroring your C++ code.

    Inspired by GTSAM, handles edge cases near w ~ ±1 using a Taylor expansion.
    """
    w = jnp.array(q.w)
    v = q.vec()

    nearly_one = 1.0 - 1e-12
    nearly_negative_one = -1.0 + 1e-12

    def near_positive_one(_):
        # w ~ +1 => small rotation => use Taylor expansion
        # (8/3 - 2/3 * w) * v
        return (8.0 / 3.0 - (2.0 / 3.0) * w) * v

    def near_negative_one(_):
        # w ~ -1 => rotation near 180° => use Taylor expansion
        # (-8/3 - 2/3 * w) * v
        return (-8.0 / 3.0 - (2.0 / 3.0) * w) * v

    def general_case(_):
        # Canonicalize sign
        sign = jnp.where(w > 0.0, 1.0, -1.0)
        w_signed = sign * w
        # Angle
        theta = 2.0 * jnp.arccos(w_signed)
        # Magnitude of the imaginary part
        s = jnp.sqrt(jnp.maximum(1.0 - w**2, 0.0))
        # Wrap angle to [-pi, pi]
        theta = wrap_to_pi(theta)
        #  (theta / s) * sign * v
        # Avoid division by zero: if s = 0 => v is zero => rotation is ID or pi
        # but that corner is caught above by expansions, so hopefully not an issue.
        return jnp.where(s < 1e-12, 0.0, (theta / s)) * sign * v

    # if w > nearly_one:
    #     return near_positive_one()
    # elif w < nearly_negative_one:
    #     return near_negative_one()
    # else:
    #     return general_case()
    return jax.lax.cond(
        w > nearly_one,
        near_positive_one,
        lambda _: jax.lax.cond(
            w < nearly_negative_one,
            near_negative_one,
            general_case,
            operand=None,
        ),
        operand=None,
    ).reshape((3, 1))
