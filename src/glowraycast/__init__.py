"""
Evaluate GLOW model on the great circle passing through an origin location along a given bearing.

Classes
-------
- GLOWRaycast
- GLOWRaycastXY

Misc. Variables
---------------
- __version__
"""

from ._glowraycast import GLOWRaycast, GLOWRaycastXY

__version__ = '1.2.0'

__all__ = ['GLOWRaycast', 'GLOWRaycastXY', '__version__']
