"""Functions used to reparametrize NN outputs

"""

import torch


def reparametrize(initial_x, initial_y, t, head):
    """Reparametrize NN output to satisfy inital/boundary conditions"""
    x = initial_x + (1 - torch.exp(-t)) * head[:, 0].reshape((-1, 1))
    y = initial_y + (1 - torch.exp(-t)) * head[:, 1].reshape((-1, 1))
    px = 1 + (1 - torch.exp(-t)) * head[:, 2].reshape((-1, 1))
    py = 0 + (1 - torch.exp(-t)) * head[:, 3].reshape((-1, 1))
    return x, y, px, py
