# %%
from __future__ import annotations
from typing import SupportsFloat as Numeric
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapz
# %%
def forward(x: Numeric | np.ndarray) -> Numeric | np.ndarray:
    """Coordinate transform A -> B

    Args:
        x (Numeric | np.ndarray): Coordinate in A

    Returns:
        Numeric | np.ndarray: Coordinate in B
    """
    return np.exp(x*0.1) - 1

def backward(x: Numeric | np.ndarray) -> Numeric | np.ndarray:
    """Coordinate transform B -> A

    Args:
        x (Numeric | np.ndarray): Coordinate in B

    Returns:
        Numeric | np.ndarray: Coordinate in A
    """
    return np.log(x + 1) / 0.1

def jac_a2b_a(x: Numeric | np.ndarray) -> Numeric | np.ndarray:
    """Jacobian of transfrom from A -> B, evaluated at points in coordinate A.

    Args:
        x (Numeric | np.ndarray): Points in coordinate A.

    Returns:
        Numeric | np.ndarray: Jacobian of transform.
    """
    return (np.exp(x*0.1)*0.1)

def jac_a2b_b(x: Numeric | np.ndarray) -> Numeric | np.ndarray:
    """Jacobian of transfrom from A -> B, evaluated at points in coordinate B.

    Args:
        x (Numeric | np.ndarray): Points in coordinate B.

    Returns:
        Numeric | np.ndarray: Jacobian of transform.
    """
    x = backward(x)
    return jac_a2b_a(x)

def f(x: Numeric | np.ndarray) -> Numeric | np.ndarray:
    return x**2
# %% Coordnate A
x = np.linspace(0, 10, 20) # sys A
y = f(x) # sys A (f(x) = x^2)
plt.plot(x, y, marker='x', markerfacecolor='k')
plt.xlabel('Coordinate A')
plt.show()
# %% A -> B
z = forward(x) # non-linear transform
plt.plot(z, y, marker='x', markerfacecolor='k')
plt.xlabel('Coordinate B')
plt.show()
# %% B (uniform)
zz = np.linspace(0, 1.5, 100) # uniform grid in coordinate B
zy = np.interp(zz, z, y) # interpolate to uniform grid
plt.plot(zz, zy, marker='x', markerfacecolor='k')
plt.xlabel('Coordinate B')
plt.show()
# %% B -> A (uniform, same bounds)
zzz = backward(zz)
zzz: np.ndarray = np.linspace(zzz.min(), zzz.max(), 20)
zzy = f(zzz)
# %% Line integral in A (uniform)
print('Line integral in A (%.3f to %.3f):'%(zzz.min(), zzz.max()), trapz(zzy, zzz))
plt.plot(zzz, zzy, marker='x', markerfacecolor='k')
plt.xlabel('Coordinate A')
plt.show()
# %% Line integral in B (uniform)
print('Line integral in B (%.3f to %.3f):'%(zz.min(), zz.max()), trapz(zy / jac_a2b_b(zz), zz))
plt.plot(zz, zy / jac_a2b_a(backward(zz)), marker='x', markerfacecolor='k')
plt.xlabel('Coordinate B')
plt.show()
# %%
