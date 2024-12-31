from typing import Self, TypeVar

import jax
from jax import numpy as jnp

T = TypeVar("T", bound="VectorSpaceOperators")


class VectorSpaceOperators:
    def __add__(self: T, rhs: T) -> T:
        if self.frame != rhs.frame:
            raise ValueError(
                f"Cannot subtract {self.__class__.__name__}s in different frames: {self.frame} and {rhs.frame}"
            )
        return self.__class__(_data=self._data + rhs._data, frame=self.frame)

    def __sub__(self: T, rhs: T) -> T:
        if self.frame != rhs.frame:
            raise ValueError(
                f"Cannot subtract {self.__class__.__name__}s in different frames: {self.frame} and {rhs.frame}"
            )
        return self.__class__(_data=self._data - rhs._data, frame=self.frame)

    def __neg__(self: T) -> T:
        return self.__class__(_data=-self._data, frame=self.frame)

    def __mul__(self: T, scalar: float) -> T:
        return self.__class__(_data=self._data * scalar, frame=self.frame)

    def __rmul__(self: T, scalar: float) -> T:
        return self.__mul__(scalar)


class SpatialVector(VectorSpaceOperators):
    def __init__(self, _data: jax.Array, frame: str):
        if _data.shape != (6, 1):
            raise ValueError(
                f"Expected data to have shape (6, 1), got shape {_data.shape}"
            )
        self._data = _data
        self.frame = frame

    def tree_flatten(self):
        return (self._data,), (self.frame,)

    @classmethod
    def zero(cls, frame: str) -> Self:
        return cls(_data=jnp.zeros((6, 1)), frame=frame)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (frame,) = aux_data
        (data,) = children
        return cls(_data=data, frame=frame)


Domain = TypeVar("Domain")
Codomain = TypeVar("Codomain")


class LinearMap:
    def __init__(
        self,
        matrix: jax.Array,
        domain: type[Domain],
        codomain: type[Codomain],
    ):
        if matrix.shape != (domain.dim, codomain.dim):
            raise ValueError(
                f"Expected matrix to have shape "
                f"({domain.dim}, {codomain.dim}), got {matrix.shape}"
            )

        self.matrix = matrix
        self.domain_class = domain
        self.codomain_class = codomain

    def __matmul__(self, rhs: Domain) -> Codomain:
        if not isinstance(rhs, self.domain_class):
            raise TypeError(
                f"Expected {self.domain_class.__name__}, got {type(rhs).__name__}"
            )

        result = self.matrix @ rhs._data
        return self.codomain_class(_data=result, frame=rhs.frame)
