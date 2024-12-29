from typing import Self, Union, overload

import jax
import jax.numpy as jnp
import numpy as np

from ..spatial.screw import Twist, Wrench
from ..util.math import skew
from . import so3

# TODO: improve error and comments


def matrix_str(mat: jax.Array) -> str:
    """
    Format a 2D JAX array with an outer bracket pair and bracketed rows,
    e.g.:
        [[1.000 0.000 0.000]
        [0.000 1.000 0.000]
        [0.000 0.000 1.000]]
    """
    # Each row gets its own bracketed string: e.g. "[1.000 0.000 0.000]"
    row_strs = []
    for row in mat:
        row_str = " ".join(f"{val:.3f}" for val in row)
        row_strs.append(f"[{row_str}]")

    # Join the row strings with newlines (plus indentation),
    # then wrap them again with a bracket at the start and end.
    inner = "\n         ".join(row_strs)
    return f"[{inner}]"


@jax.tree_util.register_pytree_node_class
class SE3Transform:
    """
    Represents a rigid body transform in SE(3) that maps from one coordinate frame to anrhs.
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
        return (
            f"{__class__.__name__}(\n"
            f"    rotation=\n"
            f"        {matrix_str(self.rotation)},\n"
            f"    translation=\n"
            f"        {matrix_str(self.translation.T)},\n"
            f"    '{self.from_frame}' → '{self.to_frame}'\n"
            f")"
        )

    def tree_flatten(self):
        return (self.rotation, self.translation), (self.from_frame, self.to_frame)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (from_frame, to_frame) = aux_data
        (rotation, translation) = children
        return cls(rotation, translation, from_frame, to_frame)

    def inverse(self) -> Self:
        """Return the inverse of this transform"""
        new_rotation = self.rotation.T
        new_translation = -new_rotation @ self.translation
        return SE3Transform(
            rotation=new_rotation,
            translation=new_translation,
            from_frame=self.to_frame,
            to_frame=self.from_frame,
        )

    def __sub__(self, rhs: Self) -> Self:
        return log(self.inverse() @ rhs)

    def __matmul__(self, rhs: Self) -> Self:
        """Compose transforms: T_a_c = T_a_b @ T_b_c"""
        if self.to_frame != rhs.from_frame:
            raise ValueError(
                f"Expected rhs in frame '{self.to_frame}', got '{rhs.from_frame}'"
            )

        new_rotation = self.rotation @ rhs.rotation
        new_translation = self.rotation @ rhs.translation + self.translation
        return SE3Transform(
            new_rotation, new_translation, self.from_frame, rhs.to_frame
        )


# TODO: get rid of transpose and replace it with Coadjoint


@jax.tree_util.register_pytree_node_class
class SE3Adjoint:
    """
    Represents an adjoint transformation matrix that maps spatial vectors between frames.
    The is_transposed flag indicates whether this represents the adjoint or its
    transpose. This class will only compose with twists when is_transposed=False, and
    with wrenches when is_transposed=True.

    When is_transposed=True, from_frame and to_frame refer to mapping between _wrench_
    frames, and when is_transposed=False, they refer to mapping between _twist_ frames.
    I.e. the semantics are flipped.
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
            is_transposed: Whether this represents Ad_T'
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

    def __repr__(self):
        frame_str = f"'{self.from_frame}' → '{self.to_frame}'"
        if self._is_transposed:
            return f"{__class__.__name__}Transpose({frame_str})"
        else:
            return f"{__class__.__name__}({frame_str})"

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
            from_frame=self.from_frame,
            to_frame=self.to_frame,
            is_transposed=not self._is_transposed,
        )

    @overload
    def __matmul__(self, rhs: Twist) -> Twist: ...

    @overload
    def __matmul__(self, rhs: Wrench) -> Wrench: ...

    def __matmul__(self, rhs: Union[Twist, Wrench]) -> Union[Twist, Wrench]:
        if rhs.frame != self.from_frame:
            raise ValueError(
                f"Expected rhs in frame '{self.from_frame}', got '{rhs.frame}'"
            )

        if not self._is_transposed:
            # Adj @ twist -> twist
            if not isinstance(rhs, Twist):
                raise TypeError(
                    f"Expected Twist for Adjoint multiplication, got {type(rhs)}"
                )

            new_angular = self.rotation @ rhs.angular
            new_linear = self.rotation @ rhs.linear + self._p_cross_R @ rhs.angular

            return Twist(new_linear, new_angular, self.to_frame)
        else:
            # Adj' @ wrench -> wrench
            if not isinstance(rhs, Wrench):
                raise TypeError(
                    f"Expected Wrench for Adjoint.T multiplication, got {type(rhs)}"
                )

            new_torque = self.rotation.T @ rhs.torque - self._p_cross_R.T @ rhs.force
            new_force = self.rotation.T @ rhs.force

            return Wrench(new_force, new_torque, self.to_frame)


def log(transform: SE3Transform) -> Twist:
    angular = so3.log(transform.rotation)
    linear = so3.left_jacobian_inverse(angular) @ transform.translation
    # TODO: check frame coherence
    return Twist(linear, angular, transform.from_frame)


def adjoint(transform: SE3Transform) -> SE3Adjoint:
    return SE3Adjoint(
        translation=transform.translation,
        rotation=transform.rotation,
        from_frame=transform.from_frame,
        to_frame=transform.to_frame,
    )
