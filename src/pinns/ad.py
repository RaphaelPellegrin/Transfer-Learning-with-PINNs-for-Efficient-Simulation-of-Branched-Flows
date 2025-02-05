"""This file contains the function to perform Automatic Differentiation."""

import torch


# Code to take the derivative with respect to the input.
def diff(u: torch.Tensor, t: torch.Tensor, order: int = 1) -> torch.Tensor:
    """Performs differentiation using Automatic Differentiation (AD).

    Args:
        u:
            The input tensor to differentiate.
        t:
            The time tensor.
        order:
            The order of the derivative to compute.

    """
    # code adapted from neurodiffeq library
    # https://github.com/NeuroDiffGym/neurodiffeq/blob/master/neurodiffeq/neurodiffeq.py
    # ones = torch.ones_like(u)

    derivative = torch.cat(
        [
            torch.autograd.grad(u[:, i].sum(), t, create_graph=True)[0]
            for i in range(u.shape[1])
        ],
        1,
    )
    if derivative is None:
        print("derivative is None")
        return torch.zeros_like(t, requires_grad=True)
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

        derivative.requires_grad_()
    return derivative
