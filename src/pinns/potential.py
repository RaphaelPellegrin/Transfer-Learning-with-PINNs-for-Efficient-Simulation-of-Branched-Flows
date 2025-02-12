"""Module for computing the potential."""

import numpy as np
import pickle
from pinns.params import means_of_gaussian


# TODO: this is replacing part of the code below.
def calculate_potential(x: np.ndarray, y: np.ndarray, alpha_: float, sigma: float) -> np.ndarray:
    """Calculate potential field."""
    potential = np.zeros_like(x)
    for mu_x, mu_y in means_of_gaussian:
        r_squared = (x - mu_x)**2 + (y - mu_y)**2
        potential += -alpha_ * np.exp(-r_squared / (2 * sigma**2))
    return potential


def potential_grid(
    initial_x: float,
    final_t: float,
    alpha_: float,
    means_of_gaussian: list = means_of_gaussian,
    sigma: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Makes a grid with the potential value.

    Args:
        initial_x:
            Initial value for x
        final_t:
            final time t
        alpha_:
            constant to scale the potential
        means_of_gaussian:
            means of the Gaussians used in making the potential
        sigma:
            used when constructing the potential. Std of the Gaussian (shared)


    """
    x1: np.ndarray = np.linspace(-0.1, 1.1, 500)
    y1: np.ndarray = np.linspace(-0.1, 1.1, 500)
    x: np.ndarray
    y: np.ndarray
    x, y = np.meshgrid(x1, y1)

    # Saving the means_of_gaussian passed in
    filename: str = "Data/Means.p"
    f = open(filename, "wb")
    pickle.dump(means_of_gaussian, f)
    f.close()

    # Saving the mesh grid
    filename = f"Data/Initial_x_{str(initial_x)}_final_t_{str(final_t)}_alpha_{str(alpha_)}_grid.p"
    f = open(filename, "wb")
    pickle.dump(x, f)
    pickle.dump(y, f)
    f.close()

    potential_grid_values = 0

    for mu_x1, mu_y1 in means_of_gaussian:
        potential_grid_values += -alpha_ * np.exp(
            -(((x - mu_x1) ** 2 + (y - mu_y1) ** 2) / sigma**2) / 2
        )

    # Saving the values of potential_grid_values on the grid
    filename = f"Data/Initial_x_{str(initial_x)}_final_t_{str(final_t)}_alpha_{str(alpha_)}_grid_potential_values.p"
    f = open(filename, "wb")
    pickle.dump(potential_grid_values, f)
    f.close()

    return x1, y1, potential_grid_values