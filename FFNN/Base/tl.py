"""Transfer Learning steps

Here, we freeze the base and fine tune new heads
for new initial conditions (or new potentials).

We use the pre-trained network (see base_training.py) that
has been trained on a set of initial conditions (each ic has a 
corresponding head) and we hope that the base is generalizable to
new ic. Note that here the heads are linear:
- further work: use perturbations to do one-shot 
(see Protopapas 2024)

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/12m7eAdoMR8gPAragRlx1Jf6A_lYSCUIO

"""

import copy
import pickle
import random
import re

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim
from AD import diff
from energy import get_current_energy
from neural_network_architecture import NeuralNetwork
from params import means_of_gaussian
from plot import plot_all_TL
from reparametrize import reparametrize, unpack
from tqdm import trange

# Tell it to use GPU

if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.set_default_tensor_type(torch.cuda.DoubleTensor)
    print("Using GPU")
else:
    device = torch.device("cpu")
    torch.set_default_tensor_type(torch.DoubleTensor)
    print("No GPU found, using cpu")

LINE_W = 3
LINE_BOX_W = 2
font = {"size": 24}

plt.rc("font", **font)
# plt.rcParams['text.usetex'] = True


def perform_transfer_learning(
    network_in: NeuralNetwork,
    specify_initial_condition: bool = True,
    init_specified: float = 0.55,
    step_lr_step: int = 1000,
    step_lr_gamma: float = 0.9,
    adam_learning_rate: float = 0.01,
    sgd_momentum: float = 0,
    sgd_lr: float = 0.005,
    energy_tl_weight: int = 3,
    use_sgd_tl: bool = False,
    parametrisation: bool = True,
    tl_weighting: int = 1,
    alpha_: float = 0.1,
    sigma: float = 0.1,
    initial_x: float = 0,
    initial_px: float = 1,
    initial_py: float = 0,
    final_t: float = 1,
    width_base: int = 40,
    num_epochs_tl: int = 25000,
    grid_size: int = 400,
    number_of_heads_tl: int = 11,
    number_of_epochs: int = 5,
    path_saving_tl_model="TL/models",
    energy_conservation: bool = True,
):
    """Does TL

    Args:
        network_in:
            the trained neural network (base trained), some trained heads for a
            bundle of initial conditions (ics)
        specify_initial_condition:
            whether to specify the initial condition or to pick one at random
        init_specified:
            the initial condition (specified)
        step_lr_step:
        step_lr_gamma:
        adam_learning_rate:
            the learning rate for the adam optimizer
        sgd_momentum:
            the momentum for stochastic gradient descent
        sgd_lr:
            the learning rate for stochastic gradient descent
        energy_tl_weight:
            the weight assigned to the "energy-conservation" loss
        use_sgd_tl:
        parametrisation:
            whether the output of the NN is parametrized to satisfy the boundary
            conditions exactly or whether that should be a component of the loss.
        TL-Weighting:
        alpha_:
            constant to scale the potential
        width_base:
            the width of the base
        number_of_epochs:
            the number of epochs we train the NN for
        number_of_epochs_TL:
            the number of epochs we train the NN for the TL


        means_of_gaussian should be of the forms [[mu_x1,mu_y1],..., [mu_xn,mu_yn]]

        random_ic:
            whether we have random initial conditions within the possible y(0)
            values.
            Otherwise, we divide the [0,1] interval into (width_heads-1)
            intervals and place the initial conditions for y at each end.
        scale:
            used in the scheduler
        sigma:
            used when constructing the potential. Std of the Gaussian (shared)
        initial_x:
            inital value for x(0)
        final_t:
            final time
        means_of_gaussian:
            the means used for tge Gaussians
        alpha_:
        width_heads:
            the width of each head
        grid_size:
            the number of random points (time) used in training
        number_of_heads:
            the number of heads in the original (base+head) network
        number_of_heads_tl:
            number of heads for TL
        path_saving_tl_model:
        print_legend:
        load_weights:
        energy_conservation:
            whether to add an energy conservation loss to the total loss
        norm_clipping:
    """

    trained_network_base = copy.deepcopy(network_in)
    # Set out tensor of times
    t = torch.linspace(0, final_t, grid_size, requires_grad=True).reshape(-1, 1)

    # For comparaison
    temp_loss_TL = np.inf

    loss_record_TL: np.ndarray = np.zeros(num_epochs_tl)

    ## LOADING WEIGHTS PART if path_saving_tl_model file exists and loadWeights=True
    print("We load the previous model for transfer learning")

    # checkpoint=torch.load(path_saving_tl_model)
    # https://pytorch.org/tutorials/beginner/saving_loading_models.html
    network50 = copy.deepcopy(trained_network_base)

    if not use_sgd_tl:
        optimizer50 = optim.Adam(network50.parameters(), lr=adam_learning_rate)
    if use_sgd_tl:
        optimizer50 = optim.SGD(
            network50.parameters(), momentum=sgd_momentum, lr=sgd_lr
        )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer50, step_size=step_lr_step, gamma=step_lr_gamma
    )

    # Now we are going to freeze some layers and keep training after that
    for name, param in network50.named_parameters():
        # really not the best way to do this...
        # print("param", name)
        # Use regex
        if re.match("lin1\..*", name):
            print("Freezing", name)
            param.requires_grad = False
        # Use regex
        if re.match("lin2\..*", name):
            print("Freezing", name)
            param.requires_grad = False

    # Dictionary for the initial conditions
    initial_conditions_tl_dictionary: dict = {}
    # Dictionary for the initial energy for each initial conditions
    H0_init_TL: dict = {}

    if not specify_initial_condition:
        for j in range(number_of_heads_tl):
            # Initial conditions
            initial_condition_TL = random.randint(0, 100) / 100
            print(
                "The initial condition (for y) for TL is {}".format(
                    initial_condition_TL
                )
            )
            initial_conditions_tl_dictionary[j] = initial_condition_TL
    else:
        for j in range(number_of_heads_tl):
            # Initial conditions
            initial_condition_TL = init_specified[j]
            print(
                "The initial condition (for y) for TL is {}".format(
                    initial_condition_TL
                )
            )
            initial_conditions_tl_dictionary[j] = initial_condition_TL

    # Keep track of the number of epochs
    total_epochs_TL: int = 0

    # Dictionary keeping track of the loss for each head
    losses_each_head_TL: dict = {}
    for k in range(number_of_heads_tl):
        losses_each_head_TL[k] = np.zeros(num_epochs_tl)

    # For every epoch...

    # NO Random sampling! No need to sample every epoch!
    t: torch.Tensor = torch.linspace(0, final_t, grid_size, requires_grad=True)
    t = t.reshape(-1, 1)

    optimizer50.zero_grad()

    with trange(num_epochs_tl) as tepoch_TL:
        for ne in tepoch_TL:
            tepoch_TL.set_description(f"Epoch {ne}")
            if ne > 0:
                optimizer50.zero_grad()

            # Forward pass through the network
            # this can actually be pre-computed and passed in directly to be faster
            # It is frozen so it does not need to be trained
            # TODO
            x_base_TL = network50.base(t)
            d_TL = network50.forward_tl(x_base_TL)

            # loss
            loss_TL: float = 0
            # for saving the best loss (of individual heads)
            losses_part_current_TL = {}

            # For each head...
            for l in range(number_of_heads_tl):
                # Get the current head
                head_TL = d_TL[l]
                # Get the corresponding initial condition
                initial_y_TL = initial_conditions_tl_dictionary[l]

                # Outputs
                if parametrisation:
                    x_TL, y_TL, px_TL, py_TL = reparametrize(
                        initial_x=initial_x,
                        initial_y=initial_y_TL,
                        t=t,
                        head=head_TL,
                        initial_px=1,
                        initial_py=0,
                    )
                elif not parametrisation:
                    x_TL, y_TL, px_TL, py_TL = unpack(head_TL)

                # Derivatives
                x_dot_TL = diff(x_TL, t, 1)
                y_dot_TL = diff(y_TL, t, 1)
                px_dot_TL = diff(px_TL, t, 1)
                py_dot_TL = diff(py_TL, t, 1)

                # Loss
                l1_tl = ((x_dot_TL - px_TL) ** 2).mean()
                l2_tl = ((y_dot_TL - py_TL) ** 2).mean()

                # For the other components of the loss, we need the potential V
                # and its derivatives
                ## Partial derivatives of the potential (updated below)
                partial_x_TL = 0
                partial_y_TL = 0

                ## Energy at the initial time (updated below)
                H_0_TL, H_curr_TL, partial_x_TL, partial_y_TL = get_current_energy(
                    initial_x,
                    initial_y_TL,
                    x_TL,
                    y_TL,
                    px_TL,
                    py_TL,
                    partial_x_TL,
                    partial_y_TL,
                    alpha_=alpha_,
                    sigma=sigma,
                    means_of_gaussian=means_of_gaussian,
                )

                ## We can finally set the energy for head l
                H0_init_TL[l] = H_0_TL

                # Other components of the loss
                l3_tl = ((px_dot_TL + partial_x_TL) ** 2).mean()
                l4_tl = ((py_dot_TL + partial_y_TL) ** 2).mean()

                # Nota Bene: L1,L2,L3 and L4 are Hamilton's equations

                # Initial conditions taken into consideration into the loss
                ## Position
                if parametrisation:
                    l5_tl = 0
                    L6_TL = 0
                    L7_TL = 0
                    L8_TL = 0
                elif not parametrisation:
                    l5_tl = ((x_TL[0, 0] - initial_x) ** 2) * tl_weighting
                    L6_TL = (y_TL[0, 0] - initial_y_TL) ** 2
                    ## Velocity
                    L7_TL = (px_TL[0, 0] - initial_px) ** 2
                    L8_TL = (py_TL[0, 0] - initial_py) ** 2

                # Could add the penalty that H is constant L9
                L9_TL = ((H_0_TL - H_curr_TL) ** 2).mean()
                if not energy_conservation:
                    # total loss
                    loss_TL += (
                        l1_tl + l2_tl + l3_tl + l4_tl + l5_tl + L6_TL + L7_TL + L8_TL
                    )
                    # loss for current head
                    lossl_val_TL = (
                        l1_tl + l2_tl + l3_tl + l4_tl + l5_tl + L6_TL + L7_TL + L8_TL
                    )
                if energy_conservation:
                    # total loss
                    loss_TL += (
                        l1_tl
                        + l2_tl
                        + l3_tl
                        + l4_tl
                        + l5_tl
                        + L6_TL
                        + L7_TL
                        + L8_TL
                        + energy_tl_weight * L9_TL
                    )
                    # loss for current head
                    lossl_val_TL = (
                        l1_tl
                        + l2_tl
                        + l3_tl
                        + l4_tl
                        + l5_tl
                        + L6_TL
                        + L7_TL
                        + energy_tl_weight * L9_TL
                    )

                # the loss for head l at epoch ne is stored
                losses_each_head_TL[l][ne] = lossl_val_TL

                # the loss for head l
                losses_part_current_TL[l] = lossl_val_TL

            # Backward
            loss_TL.backward(retain_graph=True)

            # Here we perform clipping
            # (source: https://stackoverflow.com/questions/54716377/how-to-do-gradient-clipping-in-pytorch)
            """
        if norm_clipping:
            # Check that this is correct
            torch.nn.utils.clip_grad_norm_(network50.parameters(), max_norm=1000)
        """

            optimizer50.step()
            scheduler.step()
            tepoch_TL.set_postfix(loss=loss_TL.item())

            # the loss at epoch ne is stored
            # print("Updating the loss", loss_TL.item())
            loss_record_TL[ne] = loss_TL.item()

            # If it is the best loss so far, we update the best loss and saved the model
            if loss_TL.item() < temp_loss_TL:
                epoch_mini_TL = ne + total_epochs_TL
                network2_TL = copy.deepcopy(network50)
                temp_loss_TL = loss_TL.item()
                individual_losses_saved_TL = losses_part_current_TL

    try:
        print(
            "The best loss (for TL) we achieved was:",
            temp_loss_TL,
            "at epoch",
            epoch_mini_TL,
        )
    except UnboundLocalError:
        print("Increase number of epochs")

    maxi_indi_TL = 0
    for g in range(number_of_heads_tl):
        if individual_losses_saved_TL[g] > maxi_indi_TL:
            maxi_indi_TL = individual_losses_saved_TL[g]
    print("The maximum of the individual losses (for TL) was {}".format(maxi_indi_TL))
    total_epochs_TL += num_epochs_tl

    ### Save network2_TL here (to train again in the next cell) ################
    torch.save(
        {
            "model_state_dict": network2_TL.state_dict(),
            "loss": temp_loss_TL,
            "epoch": epoch_mini_TL,
            "optimizer_state_dict": optimizer50.state_dict(),
            "total_epochs": total_epochs_TL,
            "initial_condition": initial_conditions_tl_dictionary,
        },
        path_saving_tl_model,
    )
    # Saving the network
    filename = f"TL/Initial_x_{str(initial_x)}_final_t_{str(final_t)}_alpha_{str(alpha_)}_width_base_{str(width_base)}_number_of_epochs_{str(number_of_epochs)}_epochsTL{str(num_epochs_tl)}_grid_size_{str(grid_size)}_Network_state_Tl.p"
    f = open(filename, "wb")
    pickle.dump(network2_TL.state_dict(), f)
    f.close()

    # Forward pass (network2 is the best network now)
    x_base_TL2 = network2_TL.base(t)
    d2_TL = network2_TL.forward_tl(x_base_TL2)

    initial_y = initial_conditions_tl_dictionary[0]
    # Saving the loss
    filename = f"TL/Initial_x_{str(initial_x)}_Initial_y_{str(initial_y)}_final_t_{str(final_t)}_alpha_{str(alpha_)}_width_base_{str(width_base)}_number_of_epochs_{str(number_of_epochs)}_epochsTL{str(num_epochs_tl)}_grid_size_{str(grid_size)}_lossTL.p"
    f = open(filename, "wb")
    pickle.dump(loss_record_TL, f)
    f.close()

    return (
        network2_TL,
        d2_TL,
        t,
        loss_record_TL,
        losses_each_head_TL,
        initial_conditions_tl_dictionary,
        H0_init_TL,
    )


################################################################################
######################################## Main ##################################
################################################################################


def main(
    number_of_epochs: int = 50000,  # this last parameter is important as it dictates which "pre-trained" base model we use
    number_of_heads: int = 11,
    number_of_heads_tl: int = 1,
    final_time: float = 1,
    width_base: int = 40,
    initial_x=0,
    initial_px=1,
    initial_py=0,
    alpha_=0.1,
    grid_size=400,
    sigma=0.1,
    parametrisation=True,
    num_epochs_tl=25000,
) -> None:
    """Does TL

    Args:



    """

    filename_saved = f"Data/Initial_x_0_final_t_1_alpha_0.1_width_base_40_number_of_epochs{number_of_epochs}_grid_size_400_network_state.pth"
    network_base = NeuralNetwork(
        width_base=40, width_heads=10, number_heads=number_of_heads, number_heads_tl=1
    )
    network_base.load_state_dict(
        torch.load(filename_saved, map_location=torch.device("cpu"))
    )

    (
        network2_TL,
        d2_TL,
        t,
        loss_record_TL,
        losses_each_head_TL,
        initial_conditions_tl_dictionary,
        H0_init_TL,
    ) = perform_transfer_learning(
        network_base,
        specify_initial_condition=True,
        init_specified=[0.55],
        step_lr_step=1000,
        step_lr_gamma=0.995,
        sgd_lr=0.025,
        use_sgd_tl=True,
        parametrisation=True,
        initial_x=initial_x,
        initial_px=initial_px,
        initial_py=initial_py,
        num_epochs_tl=num_epochs_tl,
        grid_size=400,
        number_of_heads_tl=number_of_heads_tl,
        number_of_epochs=number_of_epochs,
        energy_conservation=True,
    )

    plot_all_TL(
        number_of_epochs=num_epochs_tl,
        number_of_heads=number_of_heads_tl,
        loss_record=loss_record_TL,
        losses_each_head=losses_each_head_TL,
        network_trained=network2_TL,
        d2=d2_TL,
        parametrisation=parametrisation,
        initial_conditions_dictionary=initial_conditions_tl_dictionary,
        initial_x=initial_x,
        final_t=final_time,
        width_base=width_base,
        alpha_=alpha_,
        grid_size=grid_size,
        sigma=sigma,
        H0_init=H0_init_TL,
        times_t=t,
        print_legend=True,
    )

    # TODO
    # Need to code up the reattach closest head
    # The idea there is that for a new IC or a new potential
    # we start with a new head that is not random:
    # we use the head corresponding to the closest
    # ic or potential that has been trained with the base


if __name__ == "__main__":
    main()
