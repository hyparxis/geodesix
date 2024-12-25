import jax
from ..spatial.screw import Twist, Wrench
from ..util.math import skew
from typing import Union, overload

# TODO: change "other" to "rhs", improve error and comments


@jax.tree_util.register_pytree_node_class
class SE3Transform:
    """
    Represents a rigid body transform in SE(3) that maps from one coordinate frame to another.
    Composed of a rotation (SO(3)) and translation (R^3).
    """

    def __init__(
        self,
        rotation: jax.Array,
        translation: jax.Array,
        from_frame: str,
        to_frame: str,
    ):
        """
        Args:
            rotation: 3x3 rotation matrix
            translation: 3x1 translation vector
            from_frame: Name of source frame
            to_frame: Name of target frame
        """
        if rotation.shape != (3, 3):
            raise ValueError(
                f"Expected 3x3 rotation matrix, got shape {rotation.shape}"
            )
        if translation.shape != (3, 1):
            raise ValueError(
                f"Expected 3x1 translation vector, got shape {translation.shape}"
            )

        self.rotation = rotation
        self.translation = translation
        self.from_frame = from_frame
        self.to_frame = to_frame

    def __repr__(self):
        return f"SE3Transform(rotation={self.rotation}, translation={self.translation.T}, from_frame='{self.from_frame}', to_frame='{self.to_frame}')"

    def tree_flatten(self):
        return (self.rotation, self.translation), (self.from_frame, self.to_frame)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (from_frame, to_frame) = aux_data
        (rotation, translation) = children
        return cls(rotation, translation, from_frame, to_frame)

    def inverse(self) -> "SE3Transform":
        """Return the inverse of this transform"""
        new_rotation = self.rotation.T
        new_translation = -new_rotation @ self.translation
        return SE3Transform(
            rotation=new_rotation,
            translation=new_translation,
            from_frame=self.to_frame,
            to_frame=self.from_frame,
        )

    def __matmul__(self, other: "SE3Transform") -> "SE3Transform":
        """Compose transforms: T_a_c = T_a_b @ T_b_c"""
        if self.to_frame != other.from_frame:
            raise ValueError(
                f"Expected other in frame '{self.to_frame}', got '{other.from_frame}'"
            )

        new_rotation = self.rotation @ other.rotation
        new_translation = self.rotation @ other.translation + self.translation
        return SE3Transform(
            new_rotation, new_translation, self.from_frame, other.to_frame
        )


@jax.tree_util.register_pytree_node_class
class SE3Adjoint:
    """
    Represents an adjoint transformation matrix that maps spatial vectors between frames.
    For a rigid body transform T from frame a to frame b:
    - For twists: V_b = Adj_ba @ V_a
    - For wrenches: F_a = Adj_ba' @ F_b
    """

    def __init__(
        self,
        translation: jax.Array,
        rotation: jax.Array,
        from_frame: str,
        to_frame: str,
        is_transposed: bool = False,
    ):
        """
        Args:
            rotation: 3x3 rotation matrix R_b_a (rotates vectors from frame a to frame b)
            translation: 3x1 translation vector p_b_a (position of a origin in b coords)
            from_frame: Name of source frame ('a')
            to_frame: Name of target frame ('b')
            is_transposed: Whether this represents Ad_T^T
        """
        if rotation.shape != (3, 3):
            raise ValueError(
                f"Expected 3x3 rotation matrix, got shape {rotation.shape}"
            )
        if translation.shape != (3, 1):
            raise ValueError(
                f"Expected 3x1 translation vector, got shape {translation.shape}"
            )

        self.translation = translation
        self.rotation = rotation
        self.from_frame = from_frame
        self.to_frame = to_frame
        self._is_transposed = is_transposed

    @property
    def _p_cross_R(self):
        # Compute skew(p)*R lazily for efficiency
        return skew(self.translation) @ self.rotation

    def tree_flatten(self):
        return (self.translation, self.rotation), (
            self.from_frame,
            self.to_frame,
            self._is_transposed,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (from_frame, to_frame, is_transposed) = aux_data
        (translation, rotation) = children
        return cls(translation, rotation, from_frame, to_frame, is_transposed)

    @property
    def T(self):
        """Return the transpose of the adjoint"""
        return SE3Adjoint(
            rotation=self.rotation,
            translation=self.translation,
            from_frame=self.to_frame,  # Note: frames swap for transpose
            to_frame=self.from_frame,
            is_transposed=not self._is_transposed,
        )

    @overload
    def __matmul__(self, other: Twist) -> Twist: ...

    @overload
    def __matmul__(self, other: Wrench) -> Wrench: ...

    def __matmul__(self, other: Union[Twist, Wrench]) -> Union[Twist, Wrench]:
        if not isinstance(other, (Twist, Wrench)):
            raise TypeError(f"Expected Twist or Wrench, got {type(other)}")

        if other.frame != self.from_frame:
            raise ValueError(
                f"Expected other in frame '{self.from_frame}', got '{other.frame}'"
            )

        if not self._is_transposed:
            # Adj @ twist -> twist
            if not isinstance(other, Twist):
                raise TypeError(
                    f"Expected Twist for Adjoint multiplication, got {type(other)}"
                )

            new_angular = self.rotation @ other.angular
            new_linear = self.rotation @ other.linear + self._p_cross_R @ other.angular

            return Twist(new_linear, new_angular, self.to_frame)
        else:
            # Adj' @ wrench -> wrench
            if not isinstance(other, Wrench):
                raise TypeError(
                    f"Expected Wrench for Adjoint.T multiplication, got {type(other)}"
                )

            new_torque = (
                self.rotation.T @ other.torque - self._p_cross_R.T @ other.force
            )
            new_force = self.rotation.T @ other.force

            return Wrench(new_force, new_torque, self.to_frame)


def adjoint(transform: SE3Transform) -> SE3Adjoint:
    return SE3Adjoint(
        translation=transform.translation,
        rotation=transform.rotation,
        from_frame=transform.from_frame,
        to_frame=transform.to_frame,
    )
