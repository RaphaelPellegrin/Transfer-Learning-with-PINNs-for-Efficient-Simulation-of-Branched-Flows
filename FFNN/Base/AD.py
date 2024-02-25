""" THis file contains the function to perform Automatic Differentiation (AD)"""

import torch


# Code to take the derivative with respect to the input.
def diff(u, t, order=1):
    # code adapted from neurodiffeq library
    # https://github.com/NeuroDiffGym/neurodiffeq/blob/master/neurodiffeq/neurodiffeq.py
    """The derivative of a variable with respect to another."""
    # ones = torch.ones_like(u)

    der = torch.cat(
        [
            torch.autograd.grad(u[:, i].sum(), t, create_graph=True)[0]
            for i in range(u.shape[1])
        ],
        1,
    )
    if der is None:
        print("derivative is None")
        return torch.zeros_like(t, requires_grad=True)
    else:
        der.requires_grad_()
    for i in range(1, order):

        der = torch.cat(
            [
                torch.autograd.grad(der[:, i].sum(), t, create_graph=True)[0]
                for i in range(der.shape[1])
            ],
            1,
        )
        # print()
        if der is None:
            print("derivative is None")
            return torch.zeros_like(t, requires_grad=True)
        else:
            der.requires_grad_()
    return der
