"""
Evaluate GLOW model on the great circle passing through an origin location along a given bearing.

Classes
-------
- glow2d_geo
- glow2d_polar

Functions
---------
- geo_model
- polar_model

Misc. Variables
---------------
- __version__
"""

from ._glow2d import glow2d_geo, glow2d_polar, geo_model, polar_model

__version__ = '3.0.0'

__all__ = ['glow2d_geo', 'glow2d_polar', 'geo_model', 'polar_model', '__version__']
