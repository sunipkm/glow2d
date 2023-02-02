"""
Evaluate GLOW model on the great circle passing through an origin location along a given bearing.

Classes
-------
- glow2d_geo
- glow2d_polar

Misc. Variables
---------------
- __version__
"""

from ._glow2d import glow2d_geo, glow2d_polar

__version__ = '2.2.0'

__all__ = ['glow2d_geo', 'glow2d_polar', '__version__']
