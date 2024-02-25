""" This file is used to get the numerical integrators 

We can use these to compare to the NN output
"""

import numpy as np
from scipy.integrate import odeint


# Use below in the Scipy Solver
def ray_tracing_system(
    u, t, means_gaussian, lam: int = 1, sigma: float = 0.1, alpha_: float = 0.1
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
        muX1 = i[0]
        muY1 = i[1]
        V += -alpha_ * np.exp(-(((x - muX1) ** 2 + (y - muY1) ** 2) / sigma**2) / 2)
        Vx += (
            alpha_
            * np.exp(-(((x - muX1) ** 2 + (y - muY1) ** 2) / sigma**2) / 2)
            * (x - muX1)
            / sigma**2
        )
        Vy += (
            alpha_
            * np.exp(-(((x - muX1) ** 2 + (y - muY1) ** 2) / sigma**2) / 2)
            * (y - muY1)
            / sigma**2
        )

    # derivatives of x, y, px, py
    derivs = [px, py, -Vx, -Vy]

    return derivs


# Scipy Solver
def numerical_integrator(
    t, x0, y0, px0, py0, means_gaussian, lam=1, sigma: float = 0.1, alpha_: float = 0.1
):
    u0 = [x0, y0, px0, py0]
    # Call the ODE solver
    solPend = odeint(
        ray_tracing_system,
        u0,
        t,
        args=(
            means_gaussian,
            lam,
            sigma,
            alpha_,
        ),
    )
    xP = solPend[:, 0]
    yP = solPend[:, 1]
    pxP = solPend[:, 2]
    pyP = solPend[:, 3]
    return xP, yP, pxP, pyP
