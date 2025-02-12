"""Module for energy conservation."""

import math

import torch


def get_current_energy(
    initial_x: float,
    initial_y: float,
    x: torch.Tensor,
    y: torch.Tensor,
    px: torch.Tensor,
    py: torch.Tensor,
    partial_x: torch.Tensor,
    partial_y: torch.Tensor,
    alpha_: float,
    sigma: float,
    means_of_gaussian: list,
) -> tuple[float, float, float, float]:
    """Updates the energy.

    Args:
        initial_x:
            inital value for x(0)
        initial_y:
            initial value for y(0)
        x:
            current value for x
        y:
            current value for y
        px:
            current value for px
        py:
            current value for py
        partial_x:
            current value for partial_x
        partial_y:
            current value for partial_y
        alpha_:
            constant to scale the potential
        sigma:
            used when constructing the potential. Std of the Gaussian (shared)

    """
    initial_energy: float = 1 / 2
    current_energy: float = (px**2 + py**2) / 2

    for mu_x, mu_y in means_of_gaussian:

        # Building the potential and updating the partial derivatives
        potential = -alpha_ * torch.exp(
            -(1 / (2 * sigma**2)) * ((x - mu_x) ** 2 + (y - mu_y) ** 2)
        )
        # Partial wrt to x
        partial_x += -potential * (x - mu_x) * (1 / (sigma**2))
        # Partial wrt to y
        partial_y += -potential * (y - mu_y) * (1 / (sigma**2))

        # Updating the energy
        initial_energy += -alpha_ * math.exp(
            -(1 / (2 * sigma**2))
            * ((initial_x - mu_x) ** 2 + (initial_y - mu_y) ** 2)
        )
        current_energy += -alpha_ * torch.exp(
            -(1 / (2 * sigma**2)) * ((x - mu_x) ** 2 + (y - mu_y) ** 2)
        )

    return initial_energy, current_energy, partial_x, partial_y
