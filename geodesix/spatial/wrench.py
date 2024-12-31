from typing import Optional

import jax
from jax import numpy as jnp

from .spatial_vector import SpatialVector


class Wrench(SpatialVector):
    dim = 6

    def __init__(
        self,
        force: Optional[jax.Array] = None,
        torque: Optional[jax.Array] = None,
        *,
        frame: str,
        _data: Optional[jax.Array] = None,
    ):
        if _data is not None:
            data = _data
        elif force is not None and torque is not None:
            if force.shape != (3, 1):
                raise ValueError(
                    f"Expected force to have shape (3, 1), got {force.shape}"
                )
            if torque.shape != (3, 1):
                raise ValueError(
                    f"Expected torque to have shape (3, 1), got {torque.shape}"
                )
            data = jnp.vstack((torque, force))
        else:
            raise ValueError("Either _data or force and torque must be provided")
        super().__init__(_data=data, frame=frame)

    def __repr__(self):
        return f"{self.__class__.__name__}(force={self.force.T}, torque={self.torque.T}, frame={self.frame})"

    @property
    def force(self) -> jax.Array:
        return self._data[:3]

    @property
    def torque(self) -> jax.Array:
        return self._data[3:]
