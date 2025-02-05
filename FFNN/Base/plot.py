"""Module for plotting functions.

THe plotting function below plots:
the energy, the losses, the trajectories
"""

import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from neural_network_architecture import NeuralNetwork
from numerical_integrators import numerical_integrator
from params import means_of_gaussian
from reparametrize import reparametrize, unpack

line_w: int = 3
line_box_w: int = 2
font: dict = {"size": 24}
plt.rc("font", **font)


def potential_grid(
    initial_x: float,
    final_t: float,
    alpha_: float,
    means_of_gaussian: list = means_of_gaussian,
    sigma: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Makes a grid with the potential value

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
    filename: str = (
        f"Data/Initial_x_{str(initial_x)}_final_t_{str(final_t)}_alpha_{str(alpha_)}_grid.p"
    )
    f = open(filename, "wb")
    pickle.dump(x, f)
    pickle.dump(y, f)
    f.close()

    potential_grid_values = 0

    for i in means_of_gaussian:
        mu_x1 = i[0]
        mu_y1 = i[1]
        potential_grid_values += -alpha_ * np.exp(
            -(((x - mu_x1) ** 2 + (y - mu_y1) ** 2) / sigma**2) / 2
        )

    # Saving the values of potential_grid_values on the grid
    filename: str = (
        f"Data/Initial_x_{str(initial_x)}_final_t_{str(final_t)}_alpha_{str(alpha_)}_grid_potential_values.p"
    )
    f = open(filename, "wb")
    pickle.dump(potential_grid_values, f)
    f.close()

    return x1, y1, potential_grid_values


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
    H0_init,
    times_t,
    print_legend: bool = True,
    tl: str = "",  # for TL
) -> None:
    """Plots the trajectories

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
        H0_init:
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
        print("The theoretical energy is {}".format(H0_init[i]))
        ax[4].plot(
            t_comparaison.cpu().detach(),
            H0_init[i] * np.ones(Nt),
            linestyle=":",
            c="r",
        )
        ax[4].set_title("Energy")

        H_curr_comparaison = (px_comparaison**2 + py_comparaison**2) / 2
        for m in range(len(means_of_gaussian)):
            # Get the current means_of_gaussian
            mu_x = means_of_gaussian[m][0]
            mu_y = means_of_gaussian[m][1]

            # Updating the energy
            H_curr_comparaison += -alpha_ * torch.exp(
                -(1 / (2 * sigma**2))
                * ((x_comparaison - mu_x) ** 2 + (y_comparaison - mu_y) ** 2)
            )
        ax[4].plot(t_comparaison.cpu().detach(), H_curr_comparaison.cpu().detach())

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
def plot_all_TL(
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
    H0_init,
    times_t,
    print_legend: bool = True,
) -> None:
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
    x_base_comparaison_TL = network_trained.base(t_comparaison)
    heads_TL = network_trained.forward_tl(x_base_comparaison_TL)

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
            x0,
            initial_conditions_dictionary[i],
            px0,
            py0,
            means_of_gaussian,
            sigma=sigma,
            alpha_=alpha_,
        )
        # maximum_x, min_final, minimum_y, maximum_y  = update_min_max(x, y)
        save_file_numerical(x, y, px, py, initial_x, final_t, alpha_, tl="_TL")

        ax[1].plot(x, y, "g", linestyle=":", linewidth=line_w)

        # Comparaison
        # Get head m
        trajectoires_xy_tl = heads_TL[i]

        if parametrisation:
            print("Initial x is {}", initial_x)
            print("Initial y is {}", initial_y)
            (
                x_comparaison_tl,
                y_comparaison_TL,
                px_comparaison_tl,
                py_comparaison_TL,
            ) = reparametrize(initial_x, initial_y, t_comparaison, trajectoires_xy_tl)
            # mse:
            mse_TL = compute_mse(
                x_comparaison_tl,
                y_comparaison_TL,
                px_comparaison_tl,
                py_comparaison_TL,
                x,
                y,
                px,
                py,
                Nt,
            )
            # Should probably do a dict that saves them / save them to a file for the cluster
            print("The mse for head {} is {}".format(i, mse_TL))

        elif not parametrisation:
            (
                x_comparaison_tl,
                y_comparaison_TL,
                px_comparaison_tl,
                py_comparaison_TL,
            ) = unpack(trajectoires_xy_tl)
            mse_TL = compute_mse(
                x_comparaison_tl,
                y_comparaison_TL,
                px_comparaison_tl,
                py_comparaison_TL,
                x,
                y,
                px,
                py,
                Nt,
            )
            # Should probably do a dict that saves them / save them to a file for the cluster
            print("The mse (TL) for head {} is {}".format(i, mse_TL))
            # Is there a way to print it for every epoch like Blake? Yes, but more expensive. I
            # think Blake should also actually consider not computing it for every epoch
            # much more efficient

        # Theoretical energy
        print("The theoretical energy is {}".format(H0_init[i]))
        ax[2].plot(
            t_comparaison.cpu().detach(),
            H0_init[i] * np.ones(Nt),
            linestyle=":",
            c="r",
        )
        ax[2].set_title("Energy (TL)")

        H_curr_comparaison_TL = (px_comparaison_tl**2 + py_comparaison_TL**2) / 2
        for m in range(len(means_of_gaussian)):
            # Get the current means_of_gaussian
            mu_x = means_of_gaussian[m][0]
            mu_y = means_of_gaussian[m][1]

            # Updating the energy
            H_curr_comparaison_TL += -alpha_ * torch.exp(
                -(1 / (2 * sigma**2))
                * ((x_comparaison_tl - mu_x) ** 2 + (y_comparaison_TL - mu_y) ** 2)
            )
        ax[2].plot(t_comparaison.cpu().detach(), H_curr_comparaison_TL.cpu().detach())

    print("For TL, we had {} head".format(number_of_heads))
    x1, y1, potential_grid_values = potential_grid(
        initial_x, final_t, alpha_, means_of_gaussian, sigma
    )

    ax[1].contourf(x1, y1, potential_grid_values, levels=20, cmap="Reds_r")
    ax[1].set_xlim(-0.1, 1.1)

    filename_fig = f"TL/Initial_x_{str(initial_x)}_Initial_y_{str(initial_y)}_final_t_{str(final_t)}_alpha_{str(alpha_)}_width_base_{str(width_base)}_number_of_epochs{str(number_of_epochs)}_grid_size_{str(grid_size)}_TRAJECTORIES_TL.png"
    plt.savefig(filename_fig)


def compute_mse_(
    trajectoires_xy,
    x: torch.Tensor,
    y: torch.Tensor,
    px: torch.Tensor,
    py: torch.Tensor,
    Nt,
) -> float:
    """Returns the Mean Square Error (MSE) between the numerical
    solution and the NN solution

    Args:
        trajectoires_xy:
        x
        y
        px
        py
        Nt:

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
    # mse:
    mse = ((x_.cpu().detach().reshape((-1, 1)) - x.reshape((-1, 1))) ** 2).mean() + (
        (y_.cpu().detach().reshape((-1, 1)) - y.reshape((-1, 1))) ** 2
    ).mean()
    mse += ((px_.cpu().detach().reshape((-1, 1)) - px.reshape((-1, 1))) ** 2).mean() + (
        (py_.cpu().detach().reshape((-1, 1)) - py.reshape((-1, 1))) ** 2
    ).mean()
    mse = mse / (4 * Nt)
    return mse


# Boundaries of plot
def update_min_max(x, y):
    if x[-1] > maximum_x:
        maximum_x = x[-1]
    if x[-1] < min_final:
        min_final = x[-1]
    if min(y) < minimum_y:
        minimum_y = min(y)
    if max(y) > maximum_y:
        maximum_y = max(y)
    return maximum_x, min_final, minimum_y, maximum_y


################################################################################
################################## Saving functions ############################
################################################################################


def save_files(
    loss_head,
    head,
    m,
    initial_x: float,
    final_t: float,
    alpha_,
    width_base: int,
    number_of_epochs: int,
    grid_size: int,
    tl="",
) -> None:
    """

    Args:
        loss_head:
        head:
        m:
        initial_x:
            inital value for x(0)
        final_t:
            final time
        alpha_:
            constant to scale the potential
        width_base:
            the width of the base network
            shared by all layers in the base
        grid_size:
            the number of random points (time) used in training


    """
    # Saving the individual losses
    filename = f"Data/Head_{str(m)}_Initial_x_{str(initial_x)}_final_t_{str(final_t)}_alpha_{str(alpha_)}_width_base_{str(width_base)}_number_of_epochs{str(number_of_epochs)}_grid_size_{str(grid_size)}loss_individual{tl}.p"
    f = open(filename, "wb")
    pickle.dump(loss_head, f)
    f.close()

    # Saving the trajectories (x)
    filename = f"Data/Head_{str(m)}_Initial_x_{str(initial_x)}_final_t_{str(final_t)}_alpha_{str(alpha_)}_width_base_{str(width_base)}_number_of_epochs{str(number_of_epochs)}_grid_size_{str(grid_size)}_Trajectory_NN_x{tl}.p"
    # os.mkdir(filename)
    f = open(filename, "wb")
    pickle.dump(head.cpu().detach()[:, 0], f)
    f.close()
    # Saving the trajectories (y)
    filename = f"Data/Head_{str(m)}_Initial_x_{str(initial_x)}_final_t_{str(final_t)}_alpha_{str(alpha_)}_width_base_{str(width_base)}_number_of_epochs{str(number_of_epochs)}_grid_size_{str(grid_size)}Trajectory_NN_y{tl}.p"
    # os.mkdir(filename)
    f = open(filename, "wb")
    pickle.dump(head.cpu().detach()[:, 1], f)
    f.close()
    # Saving the trajectories (px)
    filename = f"Data/Head_{str(m)}_Initial_x_{str(initial_x)}_final_t_{str(final_t)}_alpha_{str(alpha_)}_width_base_{str(width_base)}_number_of_epochs{str(number_of_epochs)}_grid_size_{str(grid_size)}Trajectory_NN_px{tl}.p"
    # os.mkdir(filename)
    f = open(filename, "wb")
    pickle.dump(head.cpu().detach()[:, 2], f)
    f.close()
    # Saving the trajectories (py)
    filename = f"Data/Head_{str(m)}_Initial_x_{str(initial_x)}_final_t_{str(final_t)}_alpha_{str(alpha_)}_width_base_{str(width_base)}_number_of_epochs{str(number_of_epochs)}_grid_size_{str(grid_size)}Trajectory_NN_py{tl}.p"
    # os.mkdir(filename)
    f = open(filename, "wb")
    pickle.dump(head.cpu().detach()[:, 3], f)
    f.close()


def save_file_numerical(x, y, px, py, initial_x, final_t, alpha_, tl=""):
    # Saving the (numerical trajectories)
    filename = f"Data/Initial_x_{str(initial_x)}_final_t_{str(final_t)}_alpha_{str(alpha_)}_numerical_trajectories_x{tl}.p"
    f = open(filename, "wb")
    pickle.dump(x, f)
    f.close()
    # Saving the (numerical trajectories)
    filename = f"Data/Initial_x_{str(initial_x)}_final_t_{str(final_t)}_alpha_{str(alpha_)}_numerical_trajectories_y{tl}.p"
    f = open(filename, "wb")
    pickle.dump(y, f)
    f.close()
    # Saving the (numerical trajectories)
    filename = f"Data/Initial_x_{str(initial_x)}_final_t_{str(final_t)}_alpha_{str(alpha_)}_numerical_trajectories_px{tl}.p"
    f = open(filename, "wb")
    pickle.dump(px, f)
    f.close()
    # Saving the (numerical trajectories)
    filename = f"Data/Initial_x_{str(initial_x)}_final_t_{str(final_t)}_alpha_{str(alpha_)}_numerical_trajectories_py{tl}.p"
    f = open(filename, "wb")
    pickle.dump(py, f)
    f.close()
