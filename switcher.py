from typing import TYPE_CHECKING
from dataclasses import dataclass
from contextlib import contextmanager
import array_api_compat

if TYPE_CHECKING:
    from namespace import _ArrayAPINameSpace
else:
    import numpy.array_api as _ArrayAPINameSpace
from array_

class XP(enum):
    Numpy = 1
    Jax = 2

@dataclass
class _Array:
    mode: 
    @property
    def xp(self):
        if self.mode == Numpy:
            import array_api_compat.numpy as onp
            return onp 
        elif self.mode == Jax:
            import array_api_compat.pytorch as pnp
            return pnp
        
    def set_context(self, typ: XP):
        # Code to acquire resource, e.g.:
        old_mode = typ
        self.mode = typ
        try:
            yield None
        finally:
            # Code to release resource, e.g.:
            self.mode(old_mode)

