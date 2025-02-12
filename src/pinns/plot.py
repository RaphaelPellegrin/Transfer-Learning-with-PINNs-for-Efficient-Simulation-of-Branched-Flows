"""Module for plotting functions.

THe plotting function below plots:
the energy, the losses, the trajectories
"""

import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch

from pinns.neural_network_architecture import NeuralNetwork
from pinns.numerical_integrators import numerical_integrator
from pinns.params import means_of_gaussian
from pinns.reparametrize import reparametrize, unpack
from pinns.saving_functions import save_file_numerical, save_files
from pinns.mse import compute_mse, compute_mse_
from pinns.potential import potential_grid

line_w: int = 3
line_box_w: int = 2
font: dict = {"size": 24}
plt.rc("font", **font)


# TODO: move to energy.py
def compute_energy(x_comparaison: torch.Tensor, y_comparaison: torch.Tensor, px_comparaison: torch.Tensor, py_comparaison: torch.Tensor, alpha_: float, sigma: float):
    """Compute the energy of the system."""
    h_curr_comparaison = (px_comparaison**2 + py_comparaison**2) / 2
    for mu_x, mu_y in means_of_gaussian:

        # Updating the energy
        h_curr_comparaison += -alpha_ * torch.exp(
            -(1 / (2 * sigma**2))
            * ((x_comparaison - mu_x) ** 2 + (y_comparaison - mu_y) ** 2)
        )
    return h_curr_comparaison


def plot_all(
    number_of_epochs: int,
    number_of_heads: int,
    loss_record: np.ndarray,
    losses_each_head: dict,
    network_trained: NeuralNetwork,
    d2,
    parametrisation: bool,
    initial_conditions_dictionary: dict,
    initial_x: float,
    final_t: float,
    width_base: int,
    alpha_: float,
    grid_size: int,
    sigma: float,
    h0_init,
    times_t,
    print_legend: bool = True,
    tl: str = "",  # for TL
) -> None:
    """Plots the trajectories.

    Args:
        number_of_epochs:
        number_of_heads:
            the number of heads
        loss_record:
        loss_each_head:
        network_trained:
        d2:
        parametrisation:
        initial_conditions_dictionary
        initial_x:
            inital value for x(0)
        final_t:
            final time
        width_base:
            the width of the base
        alpha_:
            constant to scale the potential
        grid_size:
            the number of random points (time) used in training
        sigma:
            used when constructing the potential. Std of the Gaussian (shared)
        h0_init:
        times_t:
        print_legend:
        tl:


    """
    # Create a figure
    f, ax = plt.subplots(5, 1, figsize=(20, 80))

    times_t, _ = torch.sort(times_t)

    # Plot the loss as a fct of the number of epochs
    ax[1].loglog(range(number_of_epochs), loss_record, label=f"Total loss {tl}")
    ax[1].set_title(f"Loss {tl}")

    # Now plot the individual trajectories and the individual losses
    for i in range(number_of_heads):
        # Get head i
        head = d2[i]
        initial_y = initial_conditions_dictionary[i]
        # The loss
        loss_head = losses_each_head[i]

        #################################################################

        # Now we print the loss and the trajectory
        # We need to detach the tensors when working on GPU
        if parametrisation:
            x_, y_, px_, py_ = reparametrize(
                initial_x=initial_x, initial_y=initial_y, t=times_t, head=head
            )
            if print_legend:
                ax[0].plot(
                    x_.cpu().detach(),
                    y_.cpu().detach(),
                    alpha=0.8,
                    ls=":",
                    label=f"NN solution {tl} for {str(i + 1)} head",
                )
                ax[1].plot(
                    range(number_of_epochs),
                    loss_head,
                    alpha=0.8,
                    label=f"{str(i + 1)} component of the loss {tl}",
                )
            else:
                ax[0].plot(x_.cpu().detach(), y_.cpu().detach(), alpha=0.8, ls=":")
                ax[1].plot(range(number_of_epochs), loss_head, alpha=0.8)
        elif not parametrisation:
            if print_legend:
                ax[0].plot(
                    head.cpu().detach()[:, 0],
                    head.cpu().detach()[:, 1],
                    alpha=0.8,
                    ls=":",
                    label=f"NN solution for {str(i + 1)} head {tl}",
                )
                ax[1].plot(
                    range(number_of_epochs),
                    loss_head,
                    alpha=0.8,
                    label=f"{str(i + 1)} component of the loss {tl}",
                )
            else:
                ax[0].plot(
                    head.cpu().detach()[:, 0],
                    head.cpu().detach()[:, 1],
                    alpha=0.8,
                    ls=":",
                )
                ax[1].plot(range(number_of_epochs), loss_head, alpha=0.8)

    # define the time
    Nt = 500
    t = np.linspace(0, final_t, Nt)

    # For the comparaison between the NN solution and the numerical solution,
    # we need to have the points at the same time. So we cannot use random times
    # Set our tensor of times
    t_comparaison = torch.linspace(0, final_t, Nt, requires_grad=True).reshape(-1, 1)
    x_base_comparaison = network_trained.base(t_comparaison)
    heads_comparaison = network_trained.forward_initial(x_base_comparaison)

    # Initial positon and velocity
    x0, px0, py0 = 0, 1, 0.0

    # Maximum and mim=nimum x at final time
    # maximum_x = initial_x
    # maximum_y: int = 0
    # minimum_y: int = 0
    # min_final = np.inf

    for i in range(number_of_heads):
        print("The initial condition used is", initial_conditions_dictionary[i])
        initial_y = initial_conditions_dictionary[i]
        x, y, px, py = numerical_integrator(
            t,
            x0=x0,
            y0=initial_conditions_dictionary[i],
            px0=px0,
            py0=py0,
            means_gaussian=means_of_gaussian,
            sigma=sigma,
            alpha_=alpha_,
        )
        # maximum_x, min_final, minimum_y, maximum_y  = update_min_max(x, y)

        save_file_numerical(x, y, px, py, initial_x, final_t, alpha_)
        ax[0].plot(x, y, "g", linestyle=":", linewidth=line_w)

        # Comparaison
        # Get head m
        trajectoires_xy = heads_comparaison[i]

        if parametrisation:
            (
                x_comparaison,
                y_comparaison,
                px_comparaison,
                py_comparaison,
            ) = reparametrize(
                initial_x,
                initial_y=initial_y,
                t=t_comparaison,
                head=trajectoires_xy,
                initial_px=1,
                initial_py=0,
            )
            # MSE
            mse = compute_mse(
                x_comparaison,
                y_comparaison,
                px_comparaison,
                py_comparaison,
                x,
                y,
                px,
                py,
                Nt,
            )
            # Should probably do a dict that saves them / save them to a file for the cluster
            print("The mse for head {} is {}".format(i, mse))

            diff_x = x_comparaison.cpu().detach().reshape((-1, 1)) - x.reshape((-1, 1))
            diff_y = y_comparaison.cpu().detach().reshape((-1, 1)) - y.reshape((-1, 1))
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

        elif not parametrisation:
            mse = compute_mse_(trajectoires_xy, x, y, px, py, Nt)
            # Should probably do a dict that saves them / save them to a file for the cluster
            print("The mse for head {} is {}".format(i, mse))

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
            x_comparaison, y_comparaison, px_comparaison, py_comparaison = unpack(
                trajectoires_xy
            )

        # Theoretical energy
        print("The theoretical energy is {}".format(h0_init[i]))
        ax[4].plot(
            t_comparaison.cpu().detach(),
            h0_init[i] * np.ones(Nt),
            linestyle=":",
            c="r",
        )
        ax[4].set_title("Energy")

        h_curr_comparaison = compute_energy(x_comparaison, y_comparaison, px_comparaison, py_comparaison, alpha_, sigma)
        ax[4].plot(t_comparaison.cpu().detach(), h_curr_comparaison.cpu().detach())

    x1, y1, potential_grid_values = potential_grid(
        initial_x, final_t, alpha_, means_of_gaussian, sigma
    )
    ax[0].contourf(x1, y1, potential_grid_values, levels=20, cmap="Reds_r")

    # Make a grid, set the title and the labels
    ax[0].set_title(
        "Solutions (NN and Numerical) with the potential potential_grid_values"
    )
    ax[0].set_xlabel("$x$")
    ax[0].set_ylabel("$y$")
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

    plt.savefig("Data/Fig.png")


# Check but I think the difference with the function above is that this one just
# does not plot the diff in x and y. If that's the case
# get rid of this function and add an argument:
# plot the diff: yes or no. That would get rid of 150+ lines of code.
def plot_all_tl(
    number_of_epochs: int,
    number_of_heads: int,
    loss_record: np.ndarray,
    losses_each_head: dict,
    network_trained: NeuralNetwork,
    d2,
    parametrisation: bool,
    initial_conditions_dictionary: dict,
    initial_x: float,
    final_t: float,
    width_base: int,
    alpha_: float,
    grid_size: int,
    sigma: float,
    h0_init,
    times_t,
    print_legend: bool = True,
) -> None:
    """Plots the trajectories.

    Args:
        number_of_epochs:
            number of epochs
        number_of_heads:
            number of heads
        loss_record:
            loss record
        losses_each_head:
            losses each head
        network_trained:
            network trained
        d2:
            d2
        parametrisation:
            parametrisation
        initial_conditions_dictionary:
            initial conditions dictionary
        initial_x:
            initial x
        final_t:
            final t
        width_base:
            width base
        alpha_:
            alpha
        grid_size:
            grid size
        sigma:
            sigma
        h0_init:
            h0 init
        times_t:
            times t
        print_legend:
            print legend

    """
    # Create a figure
    f, ax = plt.subplots(3, 1, figsize=(20, 80))

    # Plot the loss as a fct of the number of epochs
    ax[0].plot(range(number_of_epochs), loss_record, label="Total loss (for TL)")
    ax[0].set_title("Loss for TL")

    # Now plot the individual trajectories and the individual losses
    for m in range(number_of_heads):
        initial_y = initial_conditions_dictionary[m]
        # Get head m
        head = d2[m]
        # The loss
        loss_head = losses_each_head[m]

        save_files(
            loss_head,
            head,
            m,
            initial_x,
            final_t,
            alpha_,
            width_base,
            number_of_epochs,
            grid_size,
            tl="_tl"
        )
        #################################################################

        if parametrisation:
            x_, y_, px_, py_ = reparametrize(initial_x, initial_y, times_t, head)
            if print_legend:
                ax[1].plot(
                    x_.cpu().detach(),
                    y_.cpu().detach(),
                    alpha=0.8,
                    ls=":",
                    label="NN solution (after TL)for {} head".format(m + 1),
                )
                ax[0].loglog(
                    range(number_of_epochs),
                    loss_head,
                    alpha=0.8,
                    label="{} component of the loss".format(m + 1),
                )
            else:
                ax[1].plot(x_.cpu().detach(), y_.cpu().detach(), alpha=0.8, ls=":")
                ax[0].loglog(range(number_of_epochs), loss_head, alpha=0.8)

        elif not parametrisation:
            # Now we print the loss and the trajectory
            # We need to detach the tensors when working on GPU
            if print_legend:
                ax[1].plot(
                    head.cpu().detach()[:, 0],
                    head.cpu().detach()[:, 1],
                    label="NN solution (after TL) for {} head".format(m + 1),
                )
                ax[0].loglog(
                    range(number_of_epochs),
                    loss_head,
                    alpha=0.8,
                    label="{} component of the loss".format(m + 1),
                )
            else:
                ax[1].plot(head.cpu().detach()[:, 0], head.cpu().detach()[:, 1])
                ax[0].loglog(range(number_of_epochs), loss_head, alpha=0.8)

    # define the time
    Nt = 500
    t = np.linspace(0, final_t, Nt)

    t_comparaison = torch.linspace(0, final_t, Nt, requires_grad=True).reshape(-1, 1)

    # For the comparaison between the NN solution and the numerical solution,
    # we need to have the points at the same time
    # Set our tensor of times
    # t_comparaison=torch.linspace(0,final_t,Nt,requires_grad=True).reshape(-1,1)
    x_base_comparaison_tl = network_trained.base(t_comparaison)
    heads_tl = network_trained.forward_tl(x_base_comparaison_tl)

    # Initial positon and velocity
    x0, px0, py0 = 0, 1, 0.0

    # Maximum and mim=nimum x at final time
    # maximum_x = initial_x
    # maximum_y: float = 0
    # minimum_y: float = 0
    # min_final = np.inf

    for i in range(number_of_heads):
        initial_y = initial_conditions_dictionary[i]
        print("The initial condition used is", initial_conditions_dictionary[i])
        x, y, px, py = numerical_integrator(
            t,
            x0,
            initial_conditions_dictionary[i],
            px0,
            py0,
            means_of_gaussian,
            sigma=sigma,
            alpha_=alpha_,
        )
        # maximum_x, min_final, minimum_y, maximum_y  = update_min_max(x, y)
        save_file_numerical(x, y, px, py, initial_x, final_t, alpha_, tl="_tl")

        ax[1].plot(x, y, "g", linestyle=":", linewidth=line_w)

        # Comparaison
        # Get head m
        trajectoires_xy_tl = heads_tl[i]

        if parametrisation:
            print("Initial x is {}", initial_x)
            print("Initial y is {}", initial_y)
            (
                x_comparaison_tl,
                y_comparaison_tl,
                px_comparaison_tl,
                py_comparaison_tl,
            ) = reparametrize(initial_x, initial_y, t_comparaison, trajectoires_xy_tl)
            # mse:
            mse_tl = compute_mse(
                x_comparaison_tl,
                y_comparaison_tl,
                px_comparaison_tl,
                py_comparaison_tl,
                x,
                y,
                px,
                py,
                Nt,
            )
            # Should probably do a dict that saves them / save them to a file for the cluster
            print("The mse for head {} is {}".format(i, mse_tl))

        elif not parametrisation:
            (
                x_comparaison_tl,
                y_comparaison_tl,
                px_comparaison_tl,
                py_comparaison_tl,
            ) = unpack(trajectoires_xy_tl)
            mse_tl = compute_mse(
                x_comparaison_tl,
                y_comparaison_tl,
                px_comparaison_tl,
                py_comparaison_tl,
                x,
                y,
                px,
                py,
                Nt,
            )
            # Should probably do a dict that saves them / save them to a file for the cluster
            print("The mse (TL) for head {} is {}".format(i, mse_tl))
            # Is there a way to print it for every epoch like Blake? Yes, but more expensive. I
            # think Blake should also actually consider not computing it for every epoch
            # much more efficient

        # Theoretical energy
        print("The theoretical energy is {}".format(h0_init[i]))
        ax[2].plot(
            t_comparaison.cpu().detach(),
            h0_init[i] * np.ones(Nt),
            linestyle=":",
            c="r",
        )
        ax[2].set_title("Energy (TL)")

        h_curr_comparaison_tl = compute_energy(x_comparaison_tl, y_comparaison_tl, px_comparaison_tl, py_comparaison_tl, alpha_, sigma)
        ax[2].plot(t_comparaison.cpu().detach(), h_curr_comparaison_tl.cpu().detach())

    print("For TL, we had {} head".format(number_of_heads))
    x1, y1, potential_grid_values = potential_grid(
        initial_x, final_t, alpha_, means_of_gaussian, sigma
    )

    ax[1].contourf(x1, y1, potential_grid_values, levels=20, cmap="Reds_r")
    ax[1].set_xlim(-0.1, 1.1)

    filename_fig = f"TL/Initial_x_{str(initial_x)}_Initial_y_{str(initial_y)}_final_t_{str(final_t)}_alpha_{str(alpha_)}_width_base_{str(width_base)}_number_of_epochs{str(number_of_epochs)}_grid_size_{str(grid_size)}_TRAJECTORIES_tl.png"
    plt.savefig(filename_fig)
