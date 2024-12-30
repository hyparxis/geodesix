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
class SE3:
    """
    T_a_b where
        b is the from_frame (frame the transformation maps from)
        a is the to_frame (frame the transformation is expressed in)
    """

    def __init__(
        self,
        translation: jax.Array,
        rotation: jax.Array,
        from_frame: str,
        to_frame: str,
    ):
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
        return cls(
            rotation=rotation,
            translation=translation,
            from_frame=from_frame,
            to_frame=to_frame,
        )

    def inverse(self) -> Self:
        """Return the inverse of this transform"""
        new_rotation = self.rotation.T
        new_translation = -new_rotation @ self.translation
        return SE3(
            rotation=new_rotation,
            translation=new_translation,
            from_frame=self.to_frame,
            to_frame=self.from_frame,
        )

    def __sub__(self, rhs: Self) -> Twist:
        return log(rhs.inverse() @ self)

    def __matmul__(self, rhs: Self) -> Self:
        """Compose transforms: T_a_c = T_a_b @ T_b_c"""
        if self.to_frame != rhs.from_frame:
            raise ValueError(
                f"Expected rhs in frame '{self.to_frame}', got '{rhs.from_frame}'"
            )

        new_rotation = self.rotation @ rhs.rotation
        new_translation = self.rotation @ rhs.translation + self.translation
        return SE3(
            rotation=new_rotation,
            translation=new_translation,
            from_frame=self.from_frame,
            to_frame=rhs.to_frame,
        )


@jax.tree_util.register_pytree_node_class
class SE3Adjoint:
    """SE(3) Adjoint operator for transforming Twist vectors.

    Adjoint of an SE(3) transform, acts on twists: (v, ω) -> (R v + [p]× R ω, R ω).
    """

    def __init__(
        self,
        translation: jax.Array,
        rotation: jax.Array,
        from_frame: str,
        to_frame: str,
    ):
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

    def __repr__(self):
        return f"SE3Adjoint('{self.from_frame}' → '{self.to_frame}')"

    @property
    def _p_cross_R(self):
        """Compute skew(p)*R lazily for efficiency."""
        return skew(self.translation) @ self.rotation

    # ----- PyTree methods -----
    def tree_flatten(self):
        children = (self.translation, self.rotation)
        aux_data = (self.from_frame, self.to_frame)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (translation, rotation) = children
        (from_frame, to_frame) = aux_data
        return cls(translation, rotation, from_frame, to_frame)

    # ----- Action on Twist -----
    def __matmul__(self, rhs: Twist) -> Twist:
        if rhs.frame != self.from_frame:
            raise ValueError(
                f"Expected Twist in frame '{self.from_frame}', got '{rhs.frame}'"
            )
        if not isinstance(rhs, Twist):
            raise TypeError(f"SE3Adjoint can only act on Twist, got {type(rhs)}")

        # Adjoint * Twist
        new_angular = self.rotation @ rhs.angular
        new_linear = self.rotation @ rhs.linear + self._p_cross_R @ rhs.angular

        return Twist(new_linear, new_angular, self.to_frame)


@jax.tree_util.register_pytree_node_class
class SE3CoAdjoint:
    """SE(3) CoAdjoint operator for transforming Wrench vectors.

    CoAdjoint of an SE(3) transform, acts on wrenches: (f, τ) -> (Rᵀ f, Rᵀ τ − Rᵀ [p]× f).
    """

    def __init__(
        self,
        translation: jax.Array,
        rotation: jax.Array,
        from_frame: str,
        to_frame: str,
    ):
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

    def __repr__(self):
        return f"SE3CoAdjoint('{self.from_frame}' → '{self.to_frame}')"

    @property
    def _p_cross_R(self):
        """Compute skew(p)*R lazily for efficiency."""
        return skew(self.translation) @ self.rotation

    # ----- PyTree methods -----
    def tree_flatten(self):
        children = (self.translation, self.rotation)
        aux_data = (self.from_frame, self.to_frame)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (translation, rotation) = children
        (from_frame, to_frame) = aux_data
        return cls(translation, rotation, from_frame, to_frame)

    # ----- Action on Wrench -----
    def __matmul__(self, rhs: Wrench) -> Wrench:
        if rhs.frame != self.from_frame:
            raise ValueError(
                f"Expected Wrench in frame '{self.from_frame}', got '{rhs.frame}'"
            )
        if not isinstance(rhs, Wrench):
            raise TypeError(f"SE3CoAdjoint can only act on Wrench, got {type(rhs)}")

        # CoAdjoint * Wrench
        # note that self.rotation.T == Rᵀ
        R_T = self.rotation.T
        new_force = R_T @ rhs.force
        new_torque = R_T @ rhs.torque - self._p_cross_R.T @ rhs.force

        return Wrench(new_force, new_torque, frame=self.to_frame)


def log(transform: SE3) -> Twist:
    angular = so3.log(transform.rotation)
    linear = so3.left_jacobian_inverse(angular) @ transform.translation
    return Twist(linear, angular, frame=transform.from_frame)


def adjoint(transform: SE3) -> SE3Adjoint:
    return SE3Adjoint(
        translation=transform.translation,
        rotation=transform.rotation,
        from_frame=transform.from_frame,
        to_frame=transform.to_frame,
    )


def coadjoint(transform: SE3) -> SE3CoAdjoint:
    return SE3CoAdjoint(
        translation=transform.translation,
        rotation=transform.rotation,
        from_frame=transform.from_frame,
        to_frame=transform.to_frame,
    )
