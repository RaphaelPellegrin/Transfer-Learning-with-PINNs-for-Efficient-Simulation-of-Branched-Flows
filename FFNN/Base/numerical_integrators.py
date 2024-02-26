""" This file is used to get the numerical integrators 

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
    u, t, means_gaussian, sigma: float = 0.1, alpha_: float = 0.1
) -> list:
    """Returns the derivatives of the system

    We have the derivatives of:
    x,
    y,
    px,
    py

    Args:
        u:
        t:
        means_gaussian:
            the means of the Gaussian that go in the random potential
        sigma

    """
    # unpack current values of u
    x, y, px, py = u

    V = 0
    Vx = 0
    Vy = 0

    for i in means_gaussian:
        mu_x = i[0]
        mu_y = i[1]
        V += -alpha_ * np.exp(-(((x - mu_x) ** 2 + (y - mu_y) ** 2) / sigma**2) / 2)
        Vx += (
            alpha_
            * np.exp(-(((x - mu_x) ** 2 + (y - mu_y) ** 2) / sigma**2) / 2)
            * (x - mu_x)
            / sigma**2
        )
        Vy += (
            alpha_
            * np.exp(-(((x - mu_x) ** 2 + (y - mu_y) ** 2) / sigma**2) / 2)
            * (y - mu_y)
            / sigma**2
        )

    # derivatives of x, y, px, py
    derivs = [px, py, -Vx, -Vy]

    return derivs


# Scipy Solver
def numerical_integrator(
    t,
    x0: float,
    y0: float,
    px0: float,
    py0: float,
    means_gaussian,
    sigma: float = 0.1,
    alpha_: float = 0.1,
):
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
    u0 = [x0, y0, px0, py0]
    # Call the ODE solver
    solPend = odeint(
        ray_tracing_system,
        u0,
        t,
        args=(
            means_gaussian,
            sigma,
            alpha_,
        ),
    )
    xP = solPend[:, 0]
    yP = solPend[:, 1]
    pxP = solPend[:, 2]
    pyP = solPend[:, 3]
    return xP, yP, pxP, pyP
