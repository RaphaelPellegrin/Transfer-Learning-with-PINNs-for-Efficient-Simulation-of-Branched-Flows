"""Module for energy conservation"""

import math

import torch


def get_current_energy(
    initial_x: float,
    initial_y: float,
    x,
    y,
    px,
    py,
    partial_x,
    partial_y,
    alpha_: float,
    sigma: float,
    means_cell: list,
):
    """Updates the energy

    Args:
        initial_x:
            inital value for x(0)
        initial_y:
        x:
        y:
        px:
        py:
        partial_x:
        partial_y:
        alpha_:
            constant to scale the potential
        sigma:
            used when constructing the potential. Std of the Gaussian (shared)

    """
    initial_energy = 1 / 2
    current_energy = (px**2 + py**2) / 2

    for i in range(len(means_cell)):
        # Get the current means_cell
        mu_x = means_cell[i][0]
        mu_y = means_cell[i][1]

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
