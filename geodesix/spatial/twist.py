from typing import Optional

import jax
from jax import numpy as jnp

from .spatial_vector import SpatialVector


class Twist(SpatialVector):
    dim = 6

    def __init__(
        self,
        linear: Optional[jax.Array] = None,
        angular: Optional[jax.Array] = None,
        *,
        frame: str,
        _data: Optional[jax.Array] = None,
    ):
        if _data is not None:
            data = _data
        elif linear is not None and angular is not None:
            if linear.shape != (3, 1):
                raise ValueError(
                    f"Expected linear velocity to have shape (3, 1) got {linear.shape}"
                )
            if angular.shape != (3, 1):
                raise ValueError(
                    f"Expected angular velocity to have shape (3, 1), got {angular.shape}"
                )

            data = jnp.vstack((angular, linear))
        else:
            raise ValueError("Either _data or (linear, angular) must be provided")

        super().__init__(_data=data, frame=frame)

    def __repr__(self):
        return f"{self.__class__.__name__}(linear={self.linear.T}, angular={self.angular.T}, frame={self.frame})"

    @property
    def linear(self) -> jax.Array:
        return self._data[:3]

    @property
    def angular(self) -> jax.Array:
        return self._data[3:]
