""" This file is used to define the NN architectures we use in the paper"""

import torch.nn as nn


# Define the NN
class NeuralNetwork(nn.Module):
    """ """

    def __init__(
        self,
        width_base: int = 100,
        number_dims_heads: int = 100,
        depth_body: int = 4,
        number_heads: int = 1,
        number_heads_tl: int = 1,
    ):
        """width_base is the number of nodes within each layer
        depth_body is 1 minus the number of hidden layers
        N is the number of heads
        """
        super(NeuralNetwork, self).__init__()
        self.number_heads = number_heads
        self.number_heads_tl = number_heads_tl
        self.depth_body = depth_body
        self.width_base = width_base
        # Tanh activation function
        self.nl = nn.Tanh()
        self.lin1 = nn.Linear(1, width_base)
        self.lin2 = nn.ModuleList([nn.Linear(width_base, width_base)])
        self.lin2.extend(
            [nn.Linear(width_base, width_base) for i in range(depth_body - 1)]
        )
        self.lina = nn.ModuleList([nn.Linear(width_base, number_dims_heads)])
        self.lina.extend(
            [nn.Linear(width_base, number_dims_heads) for i in range(number_heads - 1)]
        )
        # 4 outputs for x,y, p_x, p_y
        self.lout1 = nn.ModuleList([nn.Linear(number_dims_heads, 4, bias=True)])
        self.lout1.extend(
            [
                nn.Linear(number_dims_heads, 4, bias=True)
                for i in range(number_heads - 1)
            ]
        )

        ### FOR TL
        self.lina_TL = nn.ModuleList([nn.Linear(width_base, number_dims_heads)])
        self.lina_TL.extend(
            [
                nn.Linear(width_base, number_dims_heads)
                for i in range(number_heads_tl - 1)
            ]
        )
        # 4 outputs for x,y, p_x, p_y
        self.lout1_TL = nn.ModuleList([nn.Linear(number_dims_heads, 4, bias=True)])
        self.lout1_TL.extend(
            [
                nn.Linear(number_dims_heads, 4, bias=True)
                for i in range(number_heads_tl - 1)
            ]
        )


    def base(self, t):
        x = self.lin1(t)
        x = self.nl(x)
        for m in range(self.depth_body):
            x = self.lin2[m](x)
            x = self.nl(x)
        return x

    # Forward for initial training pass
    def forward_initial(self, x):
        d = {}
        for n in range(self.number_heads):
            xa = self.lina[n](x)
            d[n] = self.lout1[n](xa)
        return d

    # forward for Transfer Learning
    def forward_TL(self, x):
        d = {}
        for n in range(self.number_heads_tl):
            xa = self.lina_TL[n](x)
            d[n] = self.lout1_TL[n](xa)
        return d
