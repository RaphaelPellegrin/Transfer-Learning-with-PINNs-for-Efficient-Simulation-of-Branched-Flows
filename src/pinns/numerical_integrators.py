"""This file is used to get the numerical integrators.

We can use these to compare to the NN output.

Specifically, we have Hamilton equations:

dx/dt = px
dy/dt = py
dpx/dt = -ðV/ðx
dpy/dt= - ðv/ðy

Given by the Hamiltonian: H(x,p) = (1/2)*||p||^2 +V(x)
"""

import numpy as np
from scipy.integrate import odeint


# Use below in the Scipy Solver
def ray_tracing_system(
    u: tuple[float, float, float, float],
    means_gaussian: list,
    sigma: float = 0.1,
    alpha_: float = 0.1,
) -> list[float]:
    """Returns the derivatives of the system.

    We have the derivatives of:
    x,
    y,
    px,
    py

    Args:
        u:
            current values of x, y, px, py
        means_gaussian:
            the means of the Gaussian that go in the random potential
        sigma:
            used in potential construction. Std of Gaussians.
        alpha_:
            used in potential construction. Scale for the potential

    """
    # unpack current values of u
    x, y, px, py = u

    V: float = 0
    vx: float = 0
    vy: float = 0

    for i in means_gaussian:
        mu_x = i[0]
        mu_y = i[1]
        V += -alpha_ * np.exp(-(((x - mu_x) ** 2 + (y - mu_y) ** 2) / sigma**2) / 2)
        vx += (
            alpha_
            * np.exp(-(((x - mu_x) ** 2 + (y - mu_y) ** 2) / sigma**2) / 2)
            * (x - mu_x)
            / sigma**2
        )
        vy += (
            alpha_
            * np.exp(-(((x - mu_x) ** 2 + (y - mu_y) ** 2) / sigma**2) / 2)
            * (y - mu_y)
            / sigma**2
        )

    # derivatives of x, y, px, py
    derivs = [px, py, -vx, -vy]

    return derivs


# Scipy Solver
def numerical_integrator(
    t: np.ndarray,
    x0: float,
    y0: float,
    px0: float,
    py0: float,
    means_gaussian: list,
    sigma: float = 0.1,
    alpha_: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns the solution to the branched flow system.

    The solution are obtained numerically here.
    We can use these to sanity check the output of PINNs.

    Args:
        t:
            times
        x0:
            initial condition x(0) for t=0
        y0:
            initial condition y(0) for t=0
        px0:
            initial condition px(0) for t=0 (initial velocity in x direction)
        py0:
            initial condition py(0) for t=0 (initial velocity in y direction)
        sigma:
            used in potential construction. Std of Gaussians.
        alpha_:
            used in potential construction. Scale for the potential

    """
    u0: list[float, float, float, float] = [x0, y0, px0, py0]
    # Call the ODE solver
    solPend: np.ndarray = odeint(
        ray_tracing_system,
        u0,
        t,
        args=(
            means_gaussian,
            sigma,
            alpha_,
        ),
    )
    xp: np.ndarray = solPend[:, 0]
    yp: np.ndarray = solPend[:, 1]
    pxp: np.ndarray = solPend[:, 2]
    pyp: np.ndarray = solPend[:, 3]
    return xp, yp, pxp, pyp
