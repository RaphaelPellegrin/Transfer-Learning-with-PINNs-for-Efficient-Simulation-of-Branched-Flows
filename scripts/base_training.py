"""This is the first step in training.

Here, we train the whole network (base and heads).
This is the initial phase, to train the base to be generalizable.
"""

# Imports
import copy
import os
import pickle
import random
import time

import click
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim
from tqdm import trange

from pinns.ad import diff
from pinns.energy import get_current_energy
from pinns.neural_network_architecture import NeuralNetwork
from pinns.params import means_of_gaussian
from pinns.plot import plot_all
from pinns.reparametrize import reparametrize, unpack

# Use GPU if possible
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.set_default_tensor_type(torch.cuda.DoubleTensor)
    print("Using GPU")
else:
    device = torch.device("cpu")
    torch.set_default_tensor_type(torch.DoubleTensor)
    print("No GPU found, using cpu")


line_w: int = 3
line_box_w: int = 2
font: dict = {"size": 24}
plt.rc("font", **font)


def initial_full_network_training(
    random_ic: bool = True,
    parametrisation: bool = False,
    scale: float = 0.7,
    alpha_: float = 0.1,
    sigma: float = 0.1,
    initial_x: float = 0,
    initial_px=1,
    initial_py=0,
    final_t: float = 1,
    means_of_gaussian: list = means_of_gaussian,
    width_base: int = 40,
    width_heads: int = 10,
    number_of_epochs: int = 25000,
    grid_size: int = 400,
    number_of_heads: int = 11,
    number_of_heads_tl: int = 1,
    load_weights: bool = False,
    energy_conservation: bool = False,
    norm_clipping: bool = False,
) -> NeuralNetwork:
    """Performs full training (base + heads).

    means_of_gaussian should be of the forms [[mu_x,mu_y],..., [mu_xn,mu_yn]]

    Args:
        random_ic:
            whether we have random initial conditions within the possible y(0)
            values. Otherwise, we divide the [0,1] interval into (width_heads-1)
            intervals and place the initial conditions for y at each end.
        parametrisation:
            whether the output of the NN is parametrized to satisfy the boundary
            conditions exactly or whether that should be a component of the
            loss.
        scale:
            used in the scheduler
        alpha_:
            constant to scale the potential
        sigma:
            used when constructing the potential. Std of the Gaussian (shared)
        initial_x:
            inital value for x(0)
        initial_px:
            inital value for px(0)
        initial_py:
            inital value for py(0)
        final_t:
            final time
        means_of_gaussian:
            the means used for tge Gaussians
        width_base:
            the width of the base
        width_heads:
            the width of each head
        number_of_epochs:
            the number of epochs we train the NN for
        grid_size:
            the number of random points (time) used in training
        number_of_heads:
            the number of heads
        number_of_heads_tl:
        load_weights:
            whether to load some pre-saved weights for the NN
        energy_conservation:
            whether to add an energy conservation loss to the total loss
        norm_clipping:
            whether to do norm clipping
    """

    # Set out tensor of times
    t = torch.linspace(0, final_t, grid_size, requires_grad=True).reshape(-1, 1)

    # We keep a log of the loss as a fct of the number of epochs
    loss_record = np.zeros(number_of_epochs)

    # For comparaison
    temporary_loss = np.inf

    t0_initial = time.time()

    # Set up the network
    network = NeuralNetwork(
        width_base=width_base,
        width_heads=width_heads,
        number_heads=number_of_heads,
        number_heads_tl=number_of_heads_tl,
    )
    # Make a deep copy
    network2 = copy.deepcopy(network)

    # optimizer. Would be fun to play more with this.
    optimizer = optim.Adam(network.parameters(), lr=1e-3)
    # scheduler for the optimizer
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=scale)

    # Dictionary for the initial conditions
    initial_conditions_dictionary: dict = {}
    # Dictionary for the initial energy for each initial conditions
    h0_init: dict = {}

    # Random create initial conditions
    if not load_weights:
        if random_ic:
            for j in range(number_of_heads):
                # Initial conditions
                initial_condition = random.randint(0, 100) / 100
                print("The initial condition (for y) is {}".format(initial_condition))
                initial_conditions_dictionary[j] = initial_condition
        else:
            a = np.linspace(0, 1, number_of_heads)
            for j in range(number_of_heads):
                initial_conditions_dictionary[j] = a[j]

    # Keep track of the number of epochs
    total_epochs: int = 0

    ## LOADING WEIGHTS PART if path_saving_model file exists and load_weights=True
    if load_weights is True:
        print("We loaded the previous model")
        checkpoint = torch.load(path_saving_model)
        device = torch.device("cuda")
        network.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        network.to(device)
        total_epochs += checkpoint["total_epochs"]
        initial_conditions_dictionary = checkpoint["initial_condition"]
        print("We previously trained for {} epochs".format(total_epochs))
        print(
            "The loss was:",
            checkpoint["loss"],
            "achieved at epoch",
            checkpoint["epoch"],
        )

    # Dictionary keeping track of the loss for each head
    losses_each_head: dict = {}
    for k in range(number_of_heads):
        losses_each_head[k] = np.zeros(number_of_epochs)

    # For every epoch...
    with trange(number_of_epochs) as tepoch:
        for ne in tepoch:
            tepoch.set_description(f"Epoch {ne}")
            optimizer.zero_grad()
            # Random sampling
            t = torch.rand(grid_size, requires_grad=True) * final_t
            t, _ = torch.sort(t)
            t[0] = 0
            t = t.reshape(-1, 1)
            # Forward pass through the network
            x_base = network.base(t)
            d = network.forward_initial(x_base)
            # loss
            loss = 0
            # for saving the best loss (of individual heads)
            losses_each_head_current = {}

            # For each head...
            for l in range(number_of_heads):
                # Get the current head
                head = d[l]
                # Get the corresponding initial condition
                initial_y = initial_conditions_dictionary[l]

                # Outputs
                if parametrisation:
                    x, y, px, py = reparametrize(
                        initial_x=initial_x,
                        initial_y=initial_y,
                        t=t,
                        head=head,
                        initial_px=initial_px,
                        initial_py=initial_py,
                    )
                elif not parametrisation:
                    x, y, px, py = unpack(head)
                # Derivatives
                x_dot = diff(x, t, 1)
                y_dot = diff(y, t, 1)
                px_dot = diff(px, t, 1)
                py_dot = diff(py, t, 1)

                # Loss
                l1 = ((x_dot - px) ** 2).mean()
                l2 = ((y_dot - py) ** 2).mean()

                # For the other components of the loss, we need the potential V
                # and its derivatives
                ## Partial derivatives of the potential (updated below)
                partial_x = 0
                partial_y = 0

                ## Energy at the initial time (updated below)
                ## H0=1/2-potential evaluated at (x0, y0) ie (px0**2+py0**2)/2 - potential evaluated at (x0,y0)
                ## H_curr=(px**2+py**2)/2-potential evaluated at (x,y)
                H_0, H_curr, partial_x, partial_y = get_current_energy(
                    initial_x,
                    initial_y,
                    x,
                    y,
                    px,
                    py,
                    partial_x,
                    partial_y,
                    alpha_,
                    sigma,
                    means_of_gaussian,
                )

                ## We can finally set the energy for head l
                h0_init[l] = H_0

                # Other components of the loss
                l3 = ((px_dot + partial_x) ** 2).mean()
                l4 = ((py_dot + partial_y) ** 2).mean()

                # Nota Bene: l1,l2,l3 and l4 are Hamilton's equations

                # Initial conditions taken into consideration into the loss
                ## Position
                if parametrisation:
                    l5 = 0
                    l6 = 0
                    l7 = 0
                    l8 = 0
                elif not parametrisation:
                    l5 = (x[0, 0] - initial_x) ** 2
                    l6 = (y[0, 0] - initial_y) ** 2
                    ## Velocity
                    l7 = (px[0, 0] - initial_px) ** 2
                    l8 = (py[0, 0] - initial_py) ** 2

                # Could add the penalty that H is constant l9
                l9 = ((H_0 - H_curr) ** 2).mean()
                if not energy_conservation:
                    # total loss
                    loss += l1 + l2 + l3 + l4 + l5 + l6 + l7 + l8
                    # loss for current head
                    lossl_val = l1 + l2 + l3 + l4 + l5 + l6 + l7 + l8
                if energy_conservation:
                    # total loss
                    loss += l1 + l2 + l3 + l4 + l5 + l6 + l7 + l8 + l9
                    # loss for current head
                    lossl_val = l1 + l2 + l3 + l4 + l5 + l6 + l7 + l8 + l9

                # the loss for head l at epoch ne is stored
                losses_each_head[l][ne] = lossl_val

                # the loss for head l
                losses_each_head_current[l] = lossl_val

            # Backward
            loss.backward()

            # Here we perform clipping
            # (source: https://stackoverflow.com/questions/54716377/how-to-do-gradient-clipping-in-pytorch)
            if norm_clipping:
                torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1000)

            optimizer.step()
            scheduler.step()
            tepoch.set_postfix(loss=loss.item())

            # the loss at epoch ne is stored
            loss_record[ne] = loss.item()

            # If it is the best loss so far, we update the best loss and saved the model
            if loss.item() < temporary_loss:
                epoch_mini = ne + total_epochs
                network2 = copy.deepcopy(network)
                temporary_loss = loss.item()
                individual_losses_saved = losses_each_head_current
    try:
        print("The best loss we achieved was:", temporary_loss, "at epoch", epoch_mini)
    except UnboundLocalError:
        print("Increase number of epochs")

    maxi_indi = 0
    for g in range(number_of_heads):
        if individual_losses_saved[g] > maxi_indi:
            maxi_indi = individual_losses_saved[g]
    print("The maximum of the individual losses was {}".format(maxi_indi))
    total_epochs += number_of_epochs

    # Forward pass (network2 is the best network now)
    x_base2 = network2.base(t)
    d2 = network2.forward_initial(x_base2)

    t1_initial = time.time()

    print(
        "The elapsed time (for the first, initial training) is {}".format(
            t1_initial - t0_initial
        )
    )

    return (
        network2,
        temporary_loss,
        epoch_mini,
        optimizer.state_dict(),
        total_epochs,
        initial_conditions_dictionary,
        d2,
        t,
        loss_record,
        losses_each_head,
        initial_conditions_dictionary,
        h0_init,
    )


################################################################################
######################################## Main ##################################
################################################################################

# finish setting up all the clicks with the inputs
# to initial_full_network_training and pass in
# to plot and potential grid as well


@click.command()
# @click.option(
#     "ne",
#     "number_of_epochs",
#     default=2500,
#     help="Number of epochs we use to train the NN",
# )
# @click.option("nh", "number_of_heads", default=11, help="Number of heads")
# @click.option("ft", "final_time", default=1, help="Final time")
# @click.option("wba", "width_base", default=40, help="Width of the base")
def main(
    number_of_epochs: int = 100,
    number_of_heads: int = 11,
    final_time: float = 1,
    width_base: int = 40,
):
    initial_x = 0
    alpha_ = 0.1
    grid_size = 400
    sigma = 0.1
    parametrisation = True

    (
        network_base,
        temporary_loss,
        epoch_mini,
        optimizer_state_dict,
        total_epochs,
        initial_conditions_dictionary,
        d2,
        t,
        loss_record,
        losses_each_head,
        initial_conditions_dictionary,
        h0_init,
    ) = initial_full_network_training(
        random_ic=False,
        parametrisation=parametrisation,
        energy_conservation=True,
        number_of_epochs=number_of_epochs,
        grid_size=grid_size,
        number_of_heads=number_of_heads,
        number_of_heads_tl=1,
        alpha_=alpha_,
        sigma=sigma,
    )

    ### Save network2 here (to train again in the next cell) ###################
    # Create the directory if it doesn't exist
    os.makedirs("Data", exist_ok=True)

    torch.save(
        {
            "model_state_dict": network_base.state_dict(),
            "loss": temporary_loss,
            "epoch": epoch_mini,
            "optimizer_state_dict": optimizer_state_dict,
            "total_epochs": total_epochs,
            "initial_condition": initial_conditions_dictionary,
        },
        "Data/model",
    )

    # Saving the initial conditions
    filename = f"Data/Initial_x_{str(initial_x)}_final_t_{str(final_time)}_alpha_{str(alpha_)}_width_base_{str(width_base)}_number_of_epochs_{str(number_of_epochs)}_grid_size_{str(grid_size)}_Initial_conditions.p"
    f = open(filename, "wb")
    pickle.dump(initial_conditions_dictionary, f)
    f.close()

    # Saving the network
    filename = f"Data/Initial_x_{str(initial_x)}_final_t_{str(final_time)}_alpha_{str(alpha_)}_width_base_{str(width_base)}_number_of_epochs{str(number_of_epochs)}_grid_size_{str(grid_size)}_network_state.pth"
    # os.mkdir(filename)
    f = open(filename, "wb")
    torch.save(network_base.state_dict(), f)
    f.close()

    # Saving the loss
    filename = f"Data/Initial_x_{str(initial_x)}_final_t_{str(final_time)}_alpha_{str(alpha_)}_width_base_{str(width_base)}_number_of_epochs_{str(number_of_epochs)}_grid_size_{str(grid_size)}_loss.p"
    f = open(filename, "wb")
    pickle.dump(loss_record, f)
    f.close()

    # here need to be able to get network_base,
    # d2 *can get with network_base and t)
    # t
    # initial_condition_dictionary
    # and h0_init to be able to reload fast.

    plot_all(
        number_of_epochs=number_of_epochs,
        number_of_heads=number_of_heads,
        loss_record=loss_record,
        losses_each_head=losses_each_head,
        network_trained=network_base,
        d2=d2,
        parametrisation=parametrisation,
        initial_conditions_dictionary=initial_conditions_dictionary,
        initial_x=initial_x,
        final_t=final_time,
        width_base=width_base,
        alpha_=alpha_,
        grid_size=grid_size,
        sigma=sigma,
        h0_init=h0_init,
        times_t=t,
        print_legend=True,
    )


if __name__ == "__main__":
    main()
