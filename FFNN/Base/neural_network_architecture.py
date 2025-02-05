""" This file is used to define the NN architectures we use in the paper."""

import torch.nn as nn


# Define the NN
class NeuralNetwork(nn.Module):
    """Neural network architecture."""

    def __init__(
        self,
        width_base: int = 100,
        width_heads: int = 100,
        depth_base: int = 4,
        number_heads: int = 1,
        number_heads_tl: int = 1,
    ):
        """Initialize the NeuralNetwork.

        width_base is the number of nodes within each layer
        depth_base is 1 minus the number of hidden layers
        N is the number of heads

        Nota bene: if we change the name of the layers, we have
        to remember to change the name in other scripts too
        (for freezing)

        Args:
            width_base:
                the width of the base network
                shared by all layers in the base
            width_heads:
                the width of the heads
                shared by all heads
            depth_base:
                the depth of the base
            number_heads:
                the number of heads
            number_heads_tl:
                the number of heads for tl (can be done in parallel)

        """
        super(NeuralNetwork, self).__init__()
        self.number_heads = number_heads
        self.number_heads_tl = number_heads_tl
        self.depth_base = depth_base
        self.width_base = width_base
        # Tanh activation function
        self.nl = nn.Tanh()
        # first hidden layer: input 1 to width_base
        self.lin1 = nn.Linear(1, width_base)
        # second hidden layer
        self.lin2 = nn.ModuleList([nn.Linear(width_base, width_base)])
        # subsequent hidden layers
        self.lin2.extend(
            [nn.Linear(width_base, width_base) for i in range(depth_base - 1)]
        )
        # from last hidden layer of base to head number 1
        self.lina = nn.ModuleList([nn.Linear(width_base, width_heads)])
        # extend to all heads
        self.lina.extend(
            [nn.Linear(width_base, width_heads) for i in range(number_heads - 1)]
        )
        # 4 outputs for x,y, p_x, p_y
        # technically not necessary has no non-linear activation function is applied
        # but leaving the possibility open
        self.lout1 = nn.ModuleList([nn.Linear(width_heads, 4, bias=True)])
        self.lout1.extend(
            [nn.Linear(width_heads, 4, bias=True) for i in range(number_heads - 1)]
        )

        ### FOR TL
        # from last hidden layer of base to head number 1
        self.lina_tl = nn.ModuleList([nn.Linear(width_base, width_heads)])
        # extend to all heads
        self.lina_tl.extend(
            [nn.Linear(width_base, width_heads) for i in range(number_heads_tl - 1)]
        )
        # 4 outputs for x,y, p_x, p_y
        self.lout1_tl = nn.ModuleList([nn.Linear(width_heads, 4, bias=True)])
        self.lout1_tl.extend(
            [nn.Linear(width_heads, 4, bias=True) for i in range(number_heads_tl - 1)]
        )

    def base(self, t):
        """Base network.

        Args:
            t:
                The input tensor.

        Returns:
            The output tensor.
        """
        x = self.lin1(t)
        x = self.nl(x)
        for m in range(self.depth_base):
            x = self.lin2[m](x)
            x = self.nl(x)
        return x

    # Forward for initial training pass
    def forward_initial(self, x) -> dict:
        """Forward pass for initial training.

        Args:
            x:
                The input tensor.

        Returns:
            A dictionary containing the output of the network.
        """
        d: dict = {}
        for n in range(self.number_heads):
            xa = self.lina[n](x)
            d[n] = self.lout1[n](xa)
        return d

    # forward for Transfer Learning
    def forward_tl(self, x) -> dict:
        """Forward pass for Transfer Learning.

        Args:
            x:
                The input tensor.

        Returns:
            A dictionary containing the output of the network.
        """
        d: dict = {}
        for n in range(self.number_heads_tl):
            xa = self.lina_tl[n](x)
            d[n] = self.lout1_tl[n](xa)
        return d
