"""Module for computing the Mean Square Error (MSE)."""

import torch


def compute_mse_(
    trajectoires_xy,
    x: torch.Tensor,
    y: torch.Tensor,
    px: torch.Tensor,
    py: torch.Tensor,
    Nt,
) -> float:
    """Returns the Mean Square Error (MSE).

    MSE between the numerical solution and the NN solution.

    Args:
        trajectoires_xy:
            trajectoires xy
        x:
            x
        y:
            y
        px:
            px
        py:
            py
        Nt:
            Nt

    """
    # mse:
    mse = ((trajectoires_xy.cpu().detach()[:, 0] - x) ** 2).mean() + (
        (trajectoires_xy.cpu().detach()[:, 1] - y) ** 2
    ).mean()
    mse += ((trajectoires_xy.cpu().detach()[:, 2] - px) ** 2).mean() + (
        (trajectoires_xy.cpu().detach()[:, 3] - py) ** 2
    ).mean()
    mse = mse / (4 * Nt)
    return mse


# MSE
def compute_mse(x_, y_, px_, py_, x, y, px, py, Nt) -> float:
    """Computes the Mean Square Error (MSE).

    MSE between the numerical solution and the NN solution.

    Args:
        x_:
            x_
        y_:
            y_
        px_:
            px_
        py_:
            py_
        x:
            x
        y:
            y
        px:
            px
        py:
            py
        Nt:
            Nt
    """
    # mse:
    mse = ((x_.cpu().detach().reshape((-1, 1)) - x.reshape((-1, 1))) ** 2).mean() + (
        (y_.cpu().detach().reshape((-1, 1)) - y.reshape((-1, 1))) ** 2
    ).mean()
    mse += ((px_.cpu().detach().reshape((-1, 1)) - px.reshape((-1, 1))) ** 2).mean() + (
        (py_.cpu().detach().reshape((-1, 1)) - py.reshape((-1, 1))) ** 2
    ).mean()
    mse = mse / (4 * Nt)
    return mse