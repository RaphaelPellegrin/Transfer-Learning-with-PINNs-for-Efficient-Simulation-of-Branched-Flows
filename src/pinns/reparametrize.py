"""Functions used to reparametrize NN outputs.

We use the reparametrisation from Mattheakis et al.
THe idea is to satisfy the IC exactly when t=0,
and decay the constraint exponentially in t.

"""

import torch


def reparametrize(
    initial_x: torch.Tensor,
    initial_y: torch.Tensor,
    t: torch.Tensor,
    head: torch.Tensor,
    initial_px: float = 1,
    initial_py: float = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reparametrizes NN output to satisfy inital/boundary conditions.

    Args:
        initial_x:
            initial posiiton at t=0, x(0)
        initial_y:
            initial posiiton at t=0, y(0)
        t:
            tensor of times
        head:
            head to unpack, and re-parametrise
        initial_px:
            initial velocity in the x component at t=0, px(0)
        initial_py:
            initial velocity in the y component at t=0, py(0)
    """
    x: torch.Tensor = initial_x + (1 - torch.exp(-t)) * head[:, 0].reshape((-1, 1))
    y: torch.Tensor = initial_y + (1 - torch.exp(-t)) * head[:, 1].reshape((-1, 1))
    px: torch.Tensor = initial_px + (1 - torch.exp(-t)) * head[:, 2].reshape((-1, 1))
    py: torch.Tensor = initial_py + (1 - torch.exp(-t)) * head[:, 3].reshape((-1, 1))
    return x, y, px, py


def unpack(head) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Used when we do not reparametrise.

    Args:
        head:
            head to unpack

    We just unpack the head.
    """
    x: torch.Tensor = head[:, 0].reshape((-1, 1))
    y: torch.Tensor = head[:, 1].reshape((-1, 1))
    px: torch.Tensor = head[:, 2].reshape((-1, 1))
    py: torch.Tensor = head[:, 3].reshape((-1, 1))
    return x, y, px, py
