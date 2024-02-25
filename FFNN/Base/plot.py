"""Module for plotting functions

THe plotting function below plots:
the energy, the losses, the trajectories
"""

import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch

from neural_network_architecture import NeuralNetwork
from numerical_integrators import numerical_integrator
from reparametrize import reparametrize
from params import means_cell

def potential_grid(
    initial_x : float, final_t : float, alpha_ : float, means_cell : list =means_cell, sigma : float = 0.1
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Makes a grid with the potential value

    Args:
        initial_x:
            Initial value for x
        final_t:
            final time t
        alpha_:
        means_cell:
            means of the Gaussians used in making the potential
        sigma:
            
    
    """
    x1 : np.ndarray = np.linspace(-0.1, 1.1, 500)
    y1 : np.ndarray = np.linspace(-0.1, 1.1, 500)
    x, y = np.meshgrid(x1, y1)

    # Saving the means_cell passed in
    filename = f"Means.p"
    f = open(filename, "wb")
    pickle.dump(means_cell, f)
    f.close()

    # Saving the mesh grid
    filename : str = (
        f"Initial_x_{str(initial_x)}_final_t_{str(final_t)}_alpha_{str(alpha_)}_grid.p"
    )
    f = open(filename, "wb")
    pickle.dump(x, f)
    pickle.dump(y, f)
    f.close()

    V = 0

    for i in means_cell:
        mu_x1 = i[0]
        mu_y1 = i[1]
        V += -alpha_ * np.exp(-(((x - mu_x1) ** 2 + (y - mu_y1) ** 2) / sigma**2) / 2)

    # Saving the values of V on the grid
    filename = f"Data/Initial_x_{str(initial_x)}_final_t_{str(final_t)}_alpha_{str(alpha_)}_grid_potential_values.p"
    f = open(filename, "wb")
    pickle.dump(V, f)
    f.close()

    return x1, y1, V

def plot_all(
    number_of_epochs : int,
    number_of_heads : int,
    loss_record : np.ndarray,
    losses_each_head : dict,
    network_trained: NeuralNetwork,
    d2,
    parametrisation : bool,
    initial_conditions_dictionary : dict,
    initial_x : float ,
    final_t : float ,
    width_base : int,
    alpha_ : float,
    grid_size : int,
    x1,
    y1,
    V,
    sig : float,
    H0_init,
    print_legend: bool = True,
) -> None:
    # Create a figure
    f, ax = plt.subplots(5, 1, figsize=(20, 80))

    # Make a grid, set the title and the labels
    ax[0].set_title("Solutions (NN and Numerical) with the potential V")
    ax[0].set_xlabel("$x$")
    ax[0].set_ylabel("$y$")

    # Plot the loss as a fct of the number of epochs
    ax[1].loglog(range(number_of_epochs), loss_record, label="Total loss")
    ax[1].set_title("Loss")

    # Now plot the individual trajectories and the individual losses
    for m in range(number_of_heads):
        # Get head m
        uf = d2[m]
        initial_y = initial_conditions_dictionary[m]
        # The loss
        loss_ = losses_each_head[m]

        # Saving the individual losses
        filename = f"Data/Head_{str(m)}_Initial_x_{str(initial_x)}_final_t_{str(final_t)}_alpha_{str(alpha_)}_width_base_{str(width_base)}_number_of_epochs{str(number_of_epochs)}_grid_size_{str(grid_size)}loss_individual.p"
        f = open(filename, "wb")
        pickle.dump(loss_, f)
        f.close()

        # Saving the trajectories (x)
        filename = f"Data/Head_{str(m)}_Initial_x_{str(initial_x)}_final_t_{str(final_t)}_alpha_{str(alpha_)}_width_base_{str(width_base)}_number_of_epochs{str(number_of_epochs)}_grid_size_{str(grid_size)}_Trajectory_NN_x.p"
        # os.mkdir(filename)
        f = open(filename, "wb")
        pickle.dump(uf.cpu().detach()[:, 0], f)
        f.close()
        # Saving the trajectories (y)
        filename = f"Data/Head_{str(m)}_Initial_x_{str(initial_x)}_final_t_{str(final_t)}_alpha_{str(alpha_)}_width_base_{str(width_base)}_number_of_epochs{str(number_of_epochs)}_grid_size_{str(grid_size)}Trajectory_NN_y.p"
        # os.mkdir(filename)
        f = open(filename, "wb")
        pickle.dump(uf.cpu().detach()[:, 1], f)
        f.close()
        # Saving the trajectories (px)
        filename = f"Data/Head_{str(m)}_Initial_x_{str(initial_x)}_final_t_{str(final_t)}_alpha_{str(alpha_)}_width_base_{str(width_base)}_number_of_epochs{str(number_of_epochs)}_grid_size_{str(grid_size)}Trajectory_NN_px.p"
        # os.mkdir(filename)
        f = open(filename, "wb")
        pickle.dump(uf.cpu().detach()[:, 2], f)
        f.close()
        # Saving the trajectories (py)
        filename = f"Data/Head_{str(m)}_Initial_x_{str(initial_x)}_final_t_{str(final_t)}_alpha_{str(alpha_)}_width_base_{str(width_base)}_number_of_epochs{str(number_of_epochs)}_grid_size_{str(grid_size)}Trajectory_NN_py.p"
        # os.mkdir(filename)
        f = open(filename, "wb")
        pickle.dump(uf.cpu().detach()[:, 3], f)
        f.close()
        #################################################################

        # Now we print the loss and the trajectory
        # We need to detach the tensors when working on GPU
        if parametrisation:
            x_, y_, px_, py_ = reparametrize(initial_x=initial_x, initial_y=initial_y, t, head=uf)
            if print_legend:
                ax[0].plot(
                    x_.cpu().detach(),
                    y_.cpu().detach(),
                    alpha=0.8,
                    ls=":",
                    label="NN solution for {} head".format(m + 1),
                )
                ax[1].plot(
                    range(number_of_epochs),
                    loss_,
                    alpha=0.8,
                    label="{} component of the loss".format(m + 1),
                )
            else:
                ax[0].plot(x_.cpu().detach(), y_.cpu().detach(), alpha=0.8, ls=":")
                ax[1].plot(range(number_of_epochs), loss_, alpha=0.8)
        elif not parametrisation:
            if print_legend:
                ax[0].plot(
                    uf.cpu().detach()[:, 0],
                    uf.cpu().detach()[:, 1],
                    alpha=0.8,
                    ls=":",
                    label="NN solution for {} head".format(m + 1),
                )
                ax[1].plot(
                    range(number_of_epochs),
                    loss_,
                    alpha=0.8,
                    label="{} component of the loss".format(m + 1),
                )
            else:
                ax[0].plot(
                    uf.cpu().detach()[:, 0],
                    uf.cpu().detach()[:, 1],
                    alpha=0.8,
                    ls=":",
                )
                ax[1].plot(range(number_of_epochs), loss_, alpha=0.8)

    # define the time
    Nt = 500
    t = np.linspace(0, final_t, Nt)

    # For the comparaison between the NN solution and the numerical solution,
    # we need to have the points at the same time. So we cannot use random times
    # Set our tensor of times
    t_comparaison = torch.linspace(0, final_t, Nt, requires_grad=True).reshape(-1, 1)
    x_base_comparaison = network_trained.base(t_comparaison)
    d_comparaison = network_trained.forward_initial(x_base_comparaison)

    # Initial positon and velocity
    x0, px0, py0 = 0, 1, 0.0

    # Maximum and mim=nimum x at final time
    maximum_x = initial_x
    maximum_y: int = 0
    minimum_y: int = 0
    min_final = np.inf

    for i in range(number_of_heads):
        print("The initial condition used is", initial_conditions_dictionary[i])
        initial_y = initial_conditions_dictionary[i]
        x, y, px, py = numerical_integrator(
            t, x0, initial_conditions_dictionary[i], px0, py0, means_cell, sig=sig, alpha_=alpha_
        )
        if x[-1] > maximum_x:
            maximum_x = x[-1]
        if x[-1] < min_final:
            min_final = x[-1]
        if min(y) < minimum_y:
            minimum_y = min(y)
        if max(y) > maximum_y:
            maximum_y = max(y)

        # Saving the (numerical trajectories)
        filename = f"Data/Initial_x_{str(initial_x)}_final_t_{str(final_t)}_alpha_{str(alpha_)}_numerical_trajectories_x.p"
        f = open(filename, "wb")
        pickle.dump(x, f)
        f.close()
        # Saving the (numerical trajectories)
        filename = f"Data/Initial_x_{str(initial_x)}_final_t_{str(final_t)}_alpha_{str(alpha_)}_numerical_trajectories_y.p"
        f = open(filename, "wb")
        pickle.dump(y, f)
        f.close()
        # Saving the (numerical trajectories)
        filename = f"Data/Initial_x_{str(initial_x)}_final_t_{str(final_t)}_alpha_{str(alpha_)}_numerical_trajectories_px.p"
        f = open(filename, "wb")
        pickle.dump(px, f)
        f.close()
        # Saving the (numerical trajectories)
        filename = f"Data/Initial_x_{str(initial_x)}_final_t_{str(final_t)}_alpha_{str(alpha_)}_numerical_trajectories_py.p"
        f = open(filename, "wb")
        pickle.dump(py, f)
        f.close()

        ax[0].plot(x, y, "g", linestyle=":", linewidth=lineW)

        # Comparaison
        # Get head m
        trajectoires_xy = d_comparaison[i]

        if parametrisation:
            x_ = initial_x + (1 - torch.exp(-t_comparaison)) * trajectoires_xy[
                :, 0
            ].reshape((-1, 1))
            y_ = initial_y + (1 - torch.exp(-t_comparaison)) * trajectoires_xy[
                :, 1
            ].reshape((-1, 1))
            px_ = 1 + (1 - torch.exp(-t_comparaison)) * trajectoires_xy[:, 2].reshape(
                (-1, 1)
            )
            py_ = 0 + (1 - torch.exp(-t_comparaison)) * trajectoires_xy[:, 3].reshape(
                (-1, 1)
            )

            # MSE:
            MSE = (
                (x_.cpu().detach().reshape((-1, 1)) - x.reshape((-1, 1))) ** 2
            ).mean() + (
                (y_.cpu().detach().reshape((-1, 1)) - y.reshape((-1, 1))) ** 2
            ).mean()
            MSE += (
                (px_.cpu().detach().reshape((-1, 1)) - px.reshape((-1, 1))) ** 2
            ).mean() + (
                (py_.cpu().detach().reshape((-1, 1)) - py.reshape((-1, 1))) ** 2
            ).mean()
            MSE = MSE / (4 * Nt)
            # Should probably do a dict that saves them / save them to a file for the cluster
            print("The MSE for head {} is {}".format(i, MSE))
            diff_x = x_.cpu().detach().reshape((-1, 1)) - x.reshape((-1, 1))
            diff_y = y_.cpu().detach().reshape((-1, 1)) - y.reshape((-1, 1))
            ax[2].plot(
                t_comparaison.cpu().detach().reshape((-1, 1)),
                diff_x.reshape((-1, 1)),
            )
            ax[2].set_title("Difference between NN solution and numerical solution -x ")
            # SOMETHING IS WRONG HERE
            ax[3].plot(
                t_comparaison.cpu().detach().reshape((-1, 1)),
                diff_y.reshape((-1, 1)),
            )
            ax[3].set_title(
                "Difference between NN solution and numerical solution - y "
            )

            px_comparaison = px_
            py_comparaison = py_
            x_comparaison = x_
            y_comparaison = y_

        elif not parametrisation:
            # MSE:
            MSE = ((trajectoires_xy.cpu().detach()[:, 0] - x) ** 2).mean() + (
                (trajectoires_xy.cpu().detach()[:, 1] - y) ** 2
            ).mean()
            MSE += ((trajectoires_xy.cpu().detach()[:, 2] - px) ** 2).mean() + (
                (trajectoires_xy.cpu().detach()[:, 3] - py) ** 2
            ).mean()
            MSE = MSE / (4 * Nt)
            # Should probably do a dict that saves them / save them to a file for the cluster
            print("The MSE for head {} is {}".format(i, MSE))

            ax[2].plot(
                t_comparaison.cpu().detach(),
                trajectoires_xy.cpu().detach()[:, 0] - x,
            )
            ax[2].set_title("Difference between NN solution and numerical solution -x ")
            ax[3].plot(
                t_comparaison.cpu().detach(),
                trajectoires_xy.cpu().detach()[:, 1] - y,
            )
            ax[3].set_title(
                "Difference between NN solution and numerical solution - y "
            )

            # Compute the energy along t_comparaison
            px_comparaison = trajectoires_xy[:, 2]
            py_comparaison = trajectoires_xy[:, 3]

            px_comparaison = px_comparaison.reshape((-1, 1))
            py_comparaison = py_comparaison.reshape((-1, 1))
            x_comparaison = trajectoires_xy[:, 0]
            y_comparaison = trajectoires_xy[:, 1]
            x_comparaison = x_comparaison.reshape((-1, 1))
            y_comparaison = y_comparaison.reshape((-1, 1))

        # Theoretical energy
        print("The theoretical energy is {}".format(H0_init[i]))
        ax[4].plot(
            t_comparaison.cpu().detach(),
            H0_init[i] * np.ones(Nt),
            linestyle=":",
            c="r",
        )
        ax[4].set_title("Energy")

        H_curr_comparaison = (px_comparaison**2 + py_comparaison**2) / 2
        for m in range(len(means_cell)):
            # Get the current means_cell
            mu_x = means_cell[m][0]
            mu_y = means_cell[m][1]

            # Updating the energy
            H_curr_comparaison += -alpha_ * torch.exp(
                -(1 / (2 * sig**2))
                * ((x_comparaison - mu_x) ** 2 + (y_comparaison - mu_y) ** 2)
            )
        ax[4].plot(t_comparaison.cpu().detach(), H_curr_comparaison.cpu().detach())

    ax[0].contourf(x1, y1, V, levels=20, cmap="Reds_r")
    ax[0].set_xlim(-0.1, 1.1)

    # Make a grid, set the title and the labels
    ax[1].legend()
    ax[1].set_title("Loss")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("Number of epochs")
    ax[1].set_ylabel("Loss")

    # Make a grid, set the title and the labels
    ax[2].set_title("Errors between the solutions (NN and Numerical) - x")
    ax[2].set_xlabel("$t$")
    ax[2].set_ylabel("difference in $x$")

    # Make a grid, set the title and the labels
    ax[3].set_title("Errors between the solutions (NN and Numerical) - y")
    ax[3].set_xlabel("$t$")
    ax[3].set_ylabel("difference in $y$")

    # Make a grid, set the title and the labels
    ax[4].set_title("Energy conservation")
    ax[4].set_xlabel("$t$")
    ax[4].set_ylabel("Energy")

    plt.show()
