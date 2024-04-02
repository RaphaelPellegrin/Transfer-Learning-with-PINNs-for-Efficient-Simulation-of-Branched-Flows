""" This file contains the function to perform Automatic Differentiation (AD)"""

import torch


# Code to take the derivative with respect to the input.
def diff(u, t: torch.Tensor, order=1) -> torch.Tensor:
    """Performs differentiation using Automatic Differentiation (AD)

    Args:
        u:
        t:
        order:
            the order of the derivative

    """
    # code adapted from neurodiffeq library
    # https://github.com/NeuroDiffGym/neurodiffeq/blob/master/neurodiffeq/neurodiffeq.py
    """The derivative of a variable with respect to another."""
    # ones = torch.ones_like(u)

    derivative: torch.Tensor | None = torch.cat(
        [
            torch.autograd.grad(u[:, i].sum(), t, create_graph=True)[0]
            for i in range(u.shape[1])
        ],
        1,
    )
    if derivative is None:
        print("derivative is None")
        return torch.zeros_like(t, requires_grad=True)
    else:
        derivative.requires_grad_()
    for i in range(1, order):

        derivative = torch.cat(
            [
                torch.autograd.grad(derivative[:, i].sum(), t, create_graph=True)[0]
                for i in range(derivative.shape[1])
            ],
            1,
        )
        if derivative is None:
            print("derivative is None")
            return torch.zeros_like(t, requires_grad=True)
        else:
            derivative.requires_grad_()
    return derivative
