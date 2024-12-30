import jax
from typing import Union, overload

from ..spatial.screw import Twist, Wrench


@jax.tree_util.register_pytree_node_class
class KinematicJacobian:
    """
    Represents a kinematic Jacobian J that maps a vector of joint velocities qdot (in joint space)
    to a spatial velocity (Twist) expressed in 'to_frame'.

    J is a (6 x n) matrix:
    - n is the number of joints (degrees of freedom)
    - Each column corresponds to the contribution of one joint's velocity to the spatial velocity of a body.

    The "to_frame" specifies the reference frame in which the resulting Twist is expressed.
    The domain (joint velocities) is abstract joint space, not a geometric frame.
    """

    def __init__(
        self,
        linear: jax.Array,
        angular: jax.Array,
        to_frame: str,
        is_transposed: bool = False,
    ):
        if angular.ndim != 2 or linear.ndim != 2:
            raise ValueError(
                f"Angular and linear must be 2D, got {angular.shape}, {linear.shape}"
            )
        if angular.shape[0] != 3 or linear.shape[0] != 3:
            raise ValueError(
                f"Angular and linear must have 3 rows, got {angular.shape}, {linear.shape}"
            )
        if angular.shape[1] != linear.shape[1]:
            raise ValueError(
                f"Angular and linear must have the same number of columns, got {angular.shape[1]}, {linear.shape[1]}"
            )

        self.linear = linear
        self.angular = angular
        self.to_frame = to_frame
        self._is_transposed = is_transposed

    def tree_flatten(self):
        return (self.linear, self.angular), (self.to_frame, self._is_transposed)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (to_frame, is_transposed) = aux_data
        (linear, angular) = children
        return cls(linear, angular, to_frame, is_transposed)

    def __repr__(self):
        if not self._is_transposed:
            shape_str = f"6x{self.linear.shape[1]}"
        else:
            shape_str = f"{self.linear.shape[1]}x6"
        return f"KinematicJacobian({shape_str}, to_frame='{self.to_frame}')"

    @property
    def T(self):
        """Return the transpose of the Jacobian"""
        return KinematicJacobian(
            linear=self.linear,
            angular=self.angular,
            to_frame=self.to_frame,
            is_transposed=not self._is_transposed,
        )

    @overload
    def __matmul__(self, other: jax.Array) -> Twist: ...

    @overload
    def __matmul__(self, other: Wrench) -> jax.Array: ...

    def __matmul__(self, other: Union[jax.Array, Wrench]) -> Union[Twist, jax.Array]:
        if not self._is_transposed:
            # J @ qdot -> Twist
            if not isinstance(other, jax.Array):
                raise TypeError(f"Expected jax.Array for qdot, got {type(other)}")

            if other.shape != (self.linear.shape[1], 1):
                raise ValueError(
                    f"Expected joint velocities to have shape {(self.linear.shape[1], 1)}, got {other.shape}"
                )

            return Twist(
                linear=self.linear @ other,
                angular=self.angular @ other,
                frame=self.to_frame,
            )
        else:
            # J' @ wrench -> joint_forces
            if not isinstance(other, Wrench):
                raise TypeError(f"Expected Wrench, got {type(other)}")

            if other.frame != self.to_frame:
                raise ValueError(
                    f"Expected Wrench in frame '{self.to_frame}', got '{other.frame}'"
                )

            angular_contrib = self.angular.T @ other.torque
            linear_contrib = self.linear.T @ other.force
            joint_forces = angular_contrib + linear_contrib

            return joint_forces
