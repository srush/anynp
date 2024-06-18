from typing import TYPE_CHECKING
from enum import Enum
from dataclasses import dataclass
from contextlib import contextmanager
import array_api_compat

if TYPE_CHECKING:
    from namespace import _ArrayAPINameSpace
else:
    import array_api_compat.numpy as _ArrayAPINameSpace


class XP(Enum):
    Numpy = 1
    Jax = 2
    Pytorch = 3

@dataclass
class _Array:
    mode: XP
    @property
    def xp(self):
        if self.mode == XP.Numpy:
            import array_api_compat.numpy as onp
            return onp 
        elif self.mode == XP.Pytorch:
            import jax.experimental.array_api as jnp
            return jnp
        elif self.mode == XP.Jax:
            import array_api_compat.pytorch as pnp
            return pnp
        
    @contextmanager
    def set_context(self, typ: XP):
        # Code to acquire resource, e.g.:
        old_mode = typ
        self.mode = typ
        try:
            yield None
        finally:
            # Code to release resource, e.g.:
            self.mode = old_mode

switch = _Array(XP.Numpy) 