"""
Evaluate GLOW model on the great circle passing through an origin location along a given bearing.

Classes
-------
- GLOW2D
- GLOWRaycast
- GLOWRaycastXY

Misc. Variables
---------------
- __version__
"""

from ._glowraycast import GLOW2D, GLOWRaycast, GLOWRaycastXY

__version__ = '2.0.0a'

__all__ = ['GLOW2D', 'GLOWRaycast', 'GLOWRaycastXY', '__version__']
