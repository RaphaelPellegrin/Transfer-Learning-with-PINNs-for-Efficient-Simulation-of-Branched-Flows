# -*- coding: utf-8 -*-
"""
Paper.ipynb
Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1wh2vOyq2gVt9n6kI2lx2nwCwon6VfU4z
(now modified)




### Mounting drive
# Nota bene: the files are saved in the repo for the paper too (Loss_11_base: image, network state, terminal output, one more file)
"""
from google.colab import drive
drive.mount('/content/gdrive')

"""## Imports"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import copy
import time
import math

# Use GPU if possible
if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.set_default_tensor_type(torch.cuda.DoubleTensor)
    print('Using GPU')
else:
    device = torch.device('cpu')
    torch.set_default_tensor_type(torch.DoubleTensor)
    print('No GPU found, using cpu')

"""## AD"""

# Code to take the derivative with respect to the input.
def diff(u, t, order=1):
    # code adapted from neurodiffeq library
    # https://github.com/NeuroDiffGym/neurodiffeq/blob/master/neurodiffeq/neurodiffeq.py
    """The derivative of a variable with respect to another.
    """
    # ones = torch.ones_like(u)

    der = torch.cat([torch.autograd.grad(u[:, i].sum(), t, create_graph=True)[0] for i in range(u.shape[1])], 1)
    if der is None:
        print('derivative is None')
        return torch.zeros_like(t, requires_grad=True)
    else:
        der.requires_grad_()
    for i in range(1, order):

        der = torch.cat([torch.autograd.grad(der[:, i].sum(), t, create_graph=True)[0] for i in range(der.shape[1])], 1)
        # print()
        if der is None:
            print('derivative is None')
            return torch.zeros_like(t, requires_grad=True)
        else:
            der.requires_grad_()
    return der

"""## Defining the NN
"""

class MyNetwork_Ray_Tracing(nn.Module):
    """
    function to learn the hidden states derivatives hdot
    """
    def __init__(self, number_dims=100, number_dims_heads=100, depth_body=4,  N=1, Number_heads_TL=1):
        """ number_dims is the number of nodes within each layer
        depth_body is 1 minus the number of hidden layers
        N is the number of heads
        """
        super(MyNetwork_Ray_Tracing, self).__init__()
        self.N=N
        self.Number_heads_TL=Number_heads_TL
        self.depth_body= depth_body
        self.number_dims = number_dims
        # Tanh activation function
        self.nl = nn.Tanh()
        self.lin1 = nn.Linear(1,number_dims)
        self.lin2 = nn.ModuleList([nn.Linear(number_dims, number_dims)])
        self.lin2.extend([nn.Linear(number_dims, number_dims) for i in range(depth_body-1)])
        self.lina = nn.ModuleList([nn.Linear(number_dims, number_dims_heads)])
        self.lina.extend([nn.Linear(number_dims, number_dims_heads) for i in range(N-1)])
        # 4 outputs for x,y, p_x, p_y
        self.lout1= nn.ModuleList([nn.Linear(number_dims_heads, 4, bias=True)])
        self.lout1.extend([nn.Linear(number_dims_heads, 4, bias=True) for i in range(N-1)])

        ### FOR TL
        self.lina_TL = nn.ModuleList([nn.Linear(number_dims, number_dims_heads)])
        self.lina_TL.extend([nn.Linear(number_dims, number_dims_heads) for i in range(Number_heads_TL-1)])
        # 4 outputs for x,y, p_x, p_y
        self.lout1_TL= nn.ModuleList([nn.Linear(number_dims_heads, 4, bias=True)])
        self.lout1_TL.extend([nn.Linear(number_dims_heads, 4, bias=True) for i in range(Number_heads_TL-1)])

    def base(self, t):
        x = self.lin1(t)
        x = self.nl(x)
        for m in range(self.depth_body):
          x = self.lin2[m](x)
          x = self.nl(x)
        return x

    def forward_initial(self, x):
        d={}
        for n in range(self.N):
          xa= self.lina[n](x)
          d[n]= self.lout1[n](xa)
        return d

    def forward_TL(self, x):
        d={}
        for n in range(self.Number_heads_TL):
          xa= self.lina_TL[n](x)
          d[n]= self.lout1_TL[n](xa)
        return d

"""## Numerical solver"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import random
lineW = 3
lineBoxW=2
font = {'size'   : 24}

plt.rc('font', **font)

# Use below in the Scipy Solver   
def f_general(u, t, means_Gaussians, lam=1, sig=.1, A_=0.1):
    # unpack current values of u
    x, y, px, py = u  

    V=0
    Vx=0
    Vy=0

    A=A_

    for i in means_Gaussians:
      muX1=i[0]
      muY1=i[1]
      V+=  - A*np.exp(- (( (x-muX1)**2 + (y-muY1)**2) / sig**2)/2)
      Vx+= A*np.exp(- (( (x-muX1)**2 + (y-muY1)**2) / sig**2)/2) * (x-muX1)/sig**2 
      Vy+= A*np.exp(- (( (x-muX1)**2 + (y-muY1)**2) / sig**2)/2) * (y-muY1)/sig**2 
    
    # derivatives of x, y, px, py
    derivs = [px, py, -Vx, -Vy] 
    
    return derivs

# Scipy Solver   
def rayTracing_general(t, x0, y0, px0, py0, means_Gaussians, lam=1, sig=.1, A_=0.1):
    u0 = [x0, y0, px0, py0]
    # Call the ODE solver
    solPend = odeint(f_general, u0, t, args=(means_Gaussians, lam, sig, A_,))
    xP = solPend[:,0];    yP  = solPend[:,1];
    pxP = solPend[:,2];   pyP = solPend[:,3]
    return xP, yP, pxP, pyP

"""Setting fonts."""

font = {'size'   : 24}

plt.rc('font', **font)


"""## Experiments

### Sums of Gaussians

Here are the means for the Gaussians.
"""

means_cell=[[0.74507886, 0.3602802 ],
       [0.40147605, 0.06139579],
       [0.94162198, 0.46722697],
       [0.79110703, 0.8973808 ],
       [0.64732527, 0.07095655],
       [0.10083943, 0.31935057],
       [0.24929806, 0.60499613],
       [0.11377013, 0.42598647],
       [0.85163671, 0.26495608],
       [0.18439795, 0.31438099]]

import copy
import time
import random
import math
import pickle
from torchvision import models
from torchsummary import summary
# Regex
import re
from tqdm import trange
from google.colab import files

def N_heads_run_Gaussiann_transfer(energy_TL_weight=3, random_ic=True, parametrisation=False, max_grid_grow=400, \
                                  scale=0.7, A_=0.1, sig=.1,\
                                   initial_x=0, final_t=1, means=means_cell, \
                                   alpha_=1, width_=40, width_heads=10, \
                                   epochs_=25000, grid_size=400, \
                                   number_of_heads=11, number_of_heads_TL=1, \
                                   PATH="models", print_legend=False, \
                                   loadWeights=False, energy_conservation=False, \
                                   norm_clipping=False):
  '''
  means should be of the forms [[mu_x1,mu_y1],..., [mu_xn,mu_yn]]
  initial_x: is the (common) starting x value for our rays
  final_y: is the final time
  width_ is the width of the base
  width_heads is the width of each head
  epochs_ is the number of epochs we train the NN for 
  number_of_heads is the number of heads
  '''
  # We will time the process
  #??Access the current time
  t0=time.time()

  # Set out tensor of times
  t=torch.linspace(0,final_t,grid_size,requires_grad=True).reshape(-1,1)

  # Number of epochs
  num_epochs = epochs_

  # We keep a log of the loss as a fct of the number of epochs
  loss_log=np.zeros(num_epochs)

  # Create a figure
  f,ax=plt.subplots(5,1,figsize=(20,80))

  # For comparaison
  temp_loss=np.inf

  for i in range(1):
      t0_initial=time.time()
      # Set up the network
      network = MyNetwork_Ray_Tracing(number_dims=width_, number_dims_heads=width_heads, N=number_of_heads,  Number_heads_TL=number_of_heads_TL)
      # Make a deep copy
      network2 = copy.deepcopy(network)

      optimizer = optim.Adam(network.parameters(),lr=1e-3)
      scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=scale)
      # Dictionary for the initial conditions
      ic={}
      # Dictionary for the initial energy for each initial conditions
      H0_init={}

      # Random create initial conditions 
      if not loadWeights:
        if random_ic:
          for j in range(number_of_heads):
            # Initial conditions
            initial_condition=random.randint(0,100)/100
            print('The initial condition (for y) is {}'.format(initial_condition))
            ic[j]=initial_condition
        else:
          a=np.linspace(0,1,number_of_heads)
          for j in range(number_of_heads):
            ic[j]=a[j]

      # Keep track of the number of epochs
      total_epochs=0

      ## LOADING WEIGHTS PART if PATH file exists and loadWeights=True
      if loadWeights==True:
          print("We loaded the previous model")
          checkpoint=torch.load(PATH)
          device = torch.device("cuda")
          network.load_state_dict(checkpoint['model_state_dict'])
          optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
          network.to(device)
          total_epochs+=checkpoint['total_epochs']
          ic=checkpoint['initial_condition']
          print("We previously trained for {} epochs".format(total_epochs))
          print('The loss was:', checkpoint['loss'], 'achieved at epoch', checkpoint['epoch'])

      #??Dictionary keeping track of the loss for each head
      losses_part={}
      for k in range(number_of_heads):
        losses_part[k]=np.zeros(num_epochs)

      # For every epoch...
      with trange(num_epochs) as tepoch:
        for ne in tepoch:
            tepoch.set_description(f"Epoch {ne}")
            optimizer.zero_grad()
            # Random sampling
            grid_size_original=grid_size
            # Random sampling
            t=torch.rand(grid_size,requires_grad=True)*final_t
            t, ind=torch.sort(t)
            t[0]=0
            t=t.reshape(-1,1)
            # Forward pass through the network
            x_base = network.base(t)
            d = network.forward_initial(x_base)
            # loss
            loss=0
            # for saving the best loss (of individual heads) 
            losses_part_current={}

            # For each head...
            for l in range(number_of_heads):
              # Get the current head
              head=d[l]
              # Get the corresponding initial condition
              initial_y=ic[l]
              
              # Outputs
              if parametrisation:
                x=initial_x+(1-torch.exp(-t))*head[:,0].reshape((-1,1))
                y=initial_y+(1-torch.exp(-t))*head[:,1].reshape((-1,1))
                px=1+(1-torch.exp(-t))*head[:,2].reshape((-1,1))
                py=0+(1-torch.exp(-t))*head[:,3].reshape((-1,1))
              elif not parametrisation:
                x=head[:,0]
                y=head[:,1]
                px=head[:,2]
                py=head[:,3]
              x=x.reshape((-1,1))
              y=y.reshape((-1,1))
              px=px.reshape((-1,1))
              py=py.reshape((-1,1))
              # Derivatives
              x_dot=diff(x,t,1)
              y_dot=diff(y,t,1)
              px_dot=diff(px,t,1)
              py_dot=diff(py,t,1)

              # Loss
              L1=((x_dot-px)**2).mean()
              L2=((y_dot-py)**2).mean()

              # For the other components of the loss, we need the potential V
              # and its derivatives
              ## Partial derivatives of the potential (updated below)
              partial_x=0
              partial_y=0

              ## Energy at the initial time (updated below)
              ## H0=1/2-potential evaluated at (x0, y0) ie (px0**2+py0**2)/2 - potential evaluated at (x0,y0)
              ## H_curr=(px**2+py**2)/2-potential evaluated at (x,y)
              H_0=1/2
              H_curr=(px**2+py**2)/2

              for i in range(len(means)):
                # Get the current means
                mu_x=means[i][0]
                mu_y=means[i][1]

                # Building the potential and updating the partial derivatives
                potential=-A_*torch.exp(-(1/(2*sig**2))*((x-mu_x)**2+(y-mu_y)**2))
                # Partial wrt to x
                partial_x+=-potential*(x-mu_x)*(1/(sig**2))
                # Partial wrt to y
                partial_y+=-potential*(y-mu_y)*(1/(sig**2))

                # Updating the energy
                H_0+=-A_*math.exp(-(1/(2*sig**2))*((initial_x-mu_x)**2+(initial_y-mu_y)**2))
                H_curr+=-A_*torch.exp(-(1/(2*sig**2))*((x-mu_x)**2+(y-mu_y)**2))


              ## We can finally set the energy for head l
              H0_init[l]=H_0

              # Other components of the loss
              L3=((px_dot+partial_x)**2).mean()
              L4=((py_dot+partial_y)**2).mean()

              # Nota Bene: L1,L2,L3 and L4 are Hamilton's equations

              # Initial conditions taken into consideration into the loss
              ## Position
              if parametrisation:
                L5=0
                L6=0
                L7=0
                L8=0
              elif not parametrisation:
                L5=((x[0,0]-initial_x)**2)
                L6=((y[0,0]-initial_y)**2)
                ## Velocity
                L7=(px[0,0]-1)**2
                L8=(py[0,0]-0)**2

              # Could add the penalty that H is constant L9
              L9=((H_0-H_curr)**2).mean()
              if not energy_conservation:
                # total loss
                loss+=L1+L2+L3+L4+L5+L6+L7+L8
                # loss for current head
                lossl_val=L1+L2+L3+L4+L5+L6+L7+L8
              if energy_conservation:
                # total loss
                loss+=L1+L2+L3+L4+L5+L6+L7+L8+L9
                # loss for current head
                lossl_val=L1+L2+L3+L4+L5+L6+L7+L8+L9

              # the loss for head l at epoch ne is stored
              losses_part[l][ne]=lossl_val

              # the loss for head l
              losses_part_current[l]=lossl_val

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
            loss_log[ne]=loss.item()

            # If it is the best loss so far, we update the best loss and saved the model
            if loss.item()<temp_loss:
              epoch_mini=ne+total_epochs
              network2=copy.deepcopy(network)
              temp_loss=loss.item()
              individual_losses_saved=losses_part_current
      try:
        print('The best loss we achieved was:', temp_loss, 'at epoch', epoch_mini)
      except UnboundLocalError:
        print("Increase number of epochs")

      maxi_indi=0
      for g in range(number_of_heads):
        if individual_losses_saved[g]>maxi_indi:
          maxi_indi=individual_losses_saved[g]  
      print('The maximum of the individual losses was {}'.format(maxi_indi))
      total_epochs+=num_epochs

      ### Save network2 here (to train again in the next cell) ######################
      torch.save({'model_state_dict': network2.state_dict(), 'loss':temp_loss,  
                  'epoch':epoch_mini, 'optimizer_state_dict':optimizer.state_dict(),
                  'total_epochs': total_epochs, 'initial_condition': ic},PATH)
      ###############################################################################


      ########## Saving to a file  #####################################
      # Saving the network
      filename = 'OInitial_x_'+str(initial_x)+'final_t_'+str(final_t)+\
      'alpha_'+ str(alpha_)+'width_'+str(width_)+\
      'epochs_'+str(epochs_)+'grid_size_'+str(grid_size)+'Network_state'+'.pth'
      #os.mkdir(filename)
      f=open(filename,"wb")
      torch.save(network2.state_dict(),f)
      f.close()
      files.download(filename)
      #################################################################

      # Forward pass (network2 is the best network now)
      x_base2 = network2.base(t)
      d2 = network2.forward_initial(x_base2)

      # Plot the loss as a fct of the number of epochs
      ax[1].loglog(range(num_epochs),loss_log, label="Total loss")
      ax[1].set_title('Loss')
      

      ########## Saving to a file  #####################################
      # Saving the loss
      filename = 'OInitial_x_'+str(initial_x)+'final_t_'+str(final_t)+\
      'alpha_'+ str(alpha_)+'width_'+str(width_)+\
      'epochs_'+str(epochs_)+'grid_size_'+str(grid_size)+'loss'+'.p'
      #os.mkdir(filename)
      f=open(filename,"wb")
      pickle.dump(loss_log,f)
      f.close()
      files.download(filename)
      #################################################################

      # Now plot the individual trajectories and the individual losses
      for m in range(number_of_heads):
        # Get head m
        uf=d2[m]
        initial_y=ic[m]
        # The loss
        loss_=losses_part[m]

        ########## Saving to a file the individual losses  #####################################
        # Saving the trajectories
        filename = 'Head_'+str(m)+'Initial_x_'+str(initial_x)+'final_t_'+\
        str(final_t)+'alpha_'+ str(alpha_)+'width_'+str(width_)+\
        'epochs_'+str(epochs_)+'grid_size_'+str(grid_size)+\
        'loss_individual'+'.p'
        #os.mkdir(filename)
        f=open(filename,"wb")
        pickle.dump(loss_,f)
        f.close()
        files.download(filename)
        ########################################################################################
        
        ########## Saving to a file  #####################################
        # Saving the trajectories
        filename = 'Head_'+str(m)+'Initial_x_'+str(initial_x)+'final_t_'+\
        str(final_t)+'alpha_'+ str(alpha_)+'width_'+str(width_)+\
        'epochs_'+str(epochs_)+'grid_size_'+str(grid_size)+\
        'Trajectory_NN_x'+'.p'
        #os.mkdir(filename)
        f=open(filename,"wb")
        pickle.dump(uf.cpu().detach()[:,0],f)
        f.close()
        # files.download(filename)
        # Saving the trajectories
        filename = 'Head_'+str(m)+'Initial_x_'+str(initial_x)+'final_t_'+\
        str(final_t)+'alpha_'+ str(alpha_)+'width_'+str(width_)+\
        'epochs_'+str(epochs_)+'grid_size_'+str(grid_size)+\
        'Trajectory_NN_y'+'.p'
        #os.mkdir(filename)
        f=open(filename,"wb")
        pickle.dump(uf.cpu().detach()[:,1],f)
        f.close()
        # files.download(filename)
        # Saving the trajectories
        filename = 'Head_'+str(m)+'Initial_x_'+str(initial_x)+'final_t_'+\
        str(final_t)+'alpha_'+ str(alpha_)+'width_'+str(width_)+\
        'epochs_'+str(epochs_)+'grid_size_'+str(grid_size)+\
        'Trajectory_NN_px'+'.p'
        #os.mkdir(filename)
        f=open(filename,"wb")
        pickle.dump(uf.cpu().detach()[:,2],f)
        f.close()
        # files.download(filename)
        # Saving the trajectories
        filename = 'Head_'+str(m)+'Initial_x_'+str(initial_x)+'final_t_'+\
        str(final_t)+'alpha_'+ str(alpha_)+'width_'+str(width_)+\
        'epochs_'+str(epochs_)+'grid_size_'+str(grid_size)+'Trajectory_NN_py'+'.p'
        #os.mkdir(filename)
        f=open(filename,"wb")
        pickle.dump(uf.cpu().detach()[:,3],f)
        f.close()
        # files.download(filename)
        ################################################################# 

        # Now we print the loss and the trajectory
        # We need to detach the tensors when working on GPU
        if parametrisation:
          x_=initial_x+(1-torch.exp(-t))*uf[:,0].reshape((-1,1))
          y_=initial_y+(1-torch.exp(-t))*uf[:,1].reshape((-1,1))
          px_=1+(1-torch.exp(-t))*uf[:,2].reshape((-1,1))
          py_=0+(1-torch.exp(-t))*uf[:,3].reshape((-1,1))
          if print_legend:
            ax[0].plot(x_.cpu().detach(),y_.cpu().detach(),alpha=0.8, ls=':', label="NN solution for {} head".format(m+1))
            t_p=np.linspace(-1,1,200)
            ax[1].plot(range(num_epochs),loss_, alpha=0.8, label='{} component of the loss'.format(m+1))
          else:
            ax[0].plot(x_.cpu().detach(),y_.cpu().detach(),alpha=0.8, ls=':')
            t_p=np.linspace(-1,1,200)
            ax[1].plot(range(num_epochs),loss_, alpha=0.8)
        elif not parametrisation:
          if print_legend:
            ax[0].plot(uf.cpu().detach()[:,0],uf.cpu().detach()[:,1],alpha=0.8, ls=':', label="NN solution for {} head".format(m+1))
            t_p=np.linspace(-1,1,200)
            ax[1].plot(range(num_epochs),loss_, alpha=0.8, label='{} component of the loss'.format(m+1))
          else:
            ax[0].plot(uf.cpu().detach()[:,0],uf.cpu().detach()[:,1],alpha=0.8, ls=':')
            t_p=np.linspace(-1,1,200)
            ax[1].plot(range(num_epochs),loss_, alpha=0.8)

      t1_initial=time.time()
      print("The elapsed time (for the first, initial training) is {}".format(t1_initial-t0_initial))


      # Make a grid, set the title and the labels
      ax[0].set_title("Solutions (NN and Numerical) with the potential V")
      ax[0].set_xlabel('$x$')
      ax[0].set_ylabel('$y$')
      ######


      # Initial conditions for y
      Max=max(ic)
      Min=min(ic)

      ########## Saving ###########
      # Saving the initial conditions
      filename = 'Initial_x_'+str(initial_x)+'final_t_'+str(final_t)+'alpha_'+ str(alpha_)+'width_'+str(width_)+\
      'epochs_'+str(epochs_)+'grid_size_'+str(grid_size)+'Initial_conditions'+'.p'
      #os.mkdir(filename)
      f=open(filename,"wb")
      pickle.dump(ic,f)
      f.close()
      # files.download(filename)
      ############################# 

      # define the time
      Nt=500
      t = np.linspace(0,final_t,Nt)

      # For the comparaison between the NN solution and the numerical solution,
      # we need to have the points at the same time
      # Set our tensor of times
      t_comparaison=torch.linspace(0,final_t,Nt,requires_grad=True).reshape(-1,1)
      x_base_comparaison = network2.base(t_comparaison)
      d_comparaison = network2.forward_initial(x_base_comparaison)

      # Initial positon and velocity
      x0, px0, py0 =  0, 1, 0.; 
      # Initial y position
      Y0 = ic

      Min=0
      Max=15

      # Maximum and mim=nimum x at final time
      maximum_x=initial_x
      maximum_y=0
      minimum_y=0
      min_final=np.inf

      for i in range(number_of_heads):
          print('The initial condition used is', Y0[i])
          initial_y=Y0[i]
          x, y, px, py = rayTracing_general(t, x0, Y0[i], px0, py0, means_cell, sig=sig, A_=A_)
          if x[-1]>maximum_x:
            maximum_x=x[-1]
          if x[-1]<min_final:
            min_final=x[-1]
          if min(y)<minimum_y:
            minimum_y=min(y)
          if max(y)>maximum_y:
            maximum_y=max(y)

          ########## Saving ###########
          # Saving the (numerical trajectories)
          filename = 'Initial_x_'+str(initial_x)+'final_t_'+str(final_t)+'alpha_'+ str(alpha_)+'width_'+str(width_)+\
          'epochs_'+str(epochs_)+'grid_size_'+str(grid_size)+'Trajectories_x'+'.p'
          #os.mkdir(filename)
          f=open(filename,"wb")
          pickle.dump(x,f)
          f.close()
          #files.download(filename)
          # Saving the (numerical trajectories)
          filename = 'Initial_x_'+str(initial_x)+'final_t_'+str(final_t)+'alpha_'+ str(alpha_)+'width_'+str(width_)+\
          'epochs_'+str(epochs_)+'grid_size_'+str(grid_size)+'Trajectories_y'+'.p'
          #os.mkdir(filename)
          f=open(filename,"wb")
          pickle.dump(y,f)
          f.close()
          #files.download(filename)
          # Saving the (numerical trajectories)
          filename = 'Initial_x_'+str(initial_x)+'final_t_'+str(final_t)+'alpha_'+ str(alpha_)+'width_'+str(width_)+\
          'epochs_'+str(epochs_)+'grid_size_'+str(grid_size)+'Trajectories_px'+'.p'
          #os.mkdir(filename)
          f=open(filename,"wb")
          pickle.dump(px,f)
          f.close()
          #files.download(filename)
          # Saving the (numerical trajectories)
          filename = 'Initial_x_'+str(initial_x)+'final_t_'+str(final_t)+'alpha_'+ str(alpha_)+'width_'+str(width_)+\
          'epochs_'+str(epochs_)+'grid_size_'+str(grid_size)+'Trajectories_py'+'.p'
          #os.mkdir(filename)
          f=open(filename,"wb")
          pickle.dump(py,f)
          f.close()
          #files.download(filename)
          #############################

          ax[0].plot(x,y,'g', linestyle=':', linewidth = lineW)

          # Comparaison
          # Get head m
          trajectoires_xy=d_comparaison[i]

          if parametrisation:
            x_=initial_x+(1-torch.exp(-t_comparaison))*trajectoires_xy[:,0].reshape((-1,1))
            y_=initial_y+(1-torch.exp(-t_comparaison))*trajectoires_xy[:,1].reshape((-1,1))
            px_=1+(1-torch.exp(-t_comparaison))*trajectoires_xy[:,2].reshape((-1,1))
            py_=0+(1-torch.exp(-t_comparaison))*trajectoires_xy[:,3].reshape((-1,1))

            # MSE: 
            MSE=((x_.cpu().detach().reshape((-1,1))-x.reshape((-1,1)))**2).mean()+((y_.cpu().detach().reshape((-1,1))-y.reshape((-1,1)))**2).mean()
            MSE+=((px_.cpu().detach().reshape((-1,1))-px.reshape((-1,1)))**2).mean()+((py_.cpu().detach().reshape((-1,1))-py.reshape((-1,1)))**2).mean()
            MSE=MSE/(4*Nt)
            # Should probably do a dict that saves them / save them to a file for the cluster
            print("The MSE for head {} is {}".format(i, MSE))
            diff_x=x_.cpu().detach().reshape((-1,1))-x.reshape((-1,1))
            diff_y=y_.cpu().detach().reshape((-1,1))-y.reshape((-1,1))
            ax[2].plot(t_comparaison.cpu().detach().reshape((-1,1)),diff_x.reshape((-1,1)))
            ax[2].set_title('Difference between NN solution and numerical solution -x ')
            # SOMETHING IS WRONG HERE
            ax[3].plot(t_comparaison.cpu().detach().reshape((-1,1)),diff_y.reshape((-1,1)))
            ax[3].set_title('Difference between NN solution and numerical solution - y ')

            px_comparaison=px_
            py_comparaison=py_
            x_comparaison=x_
            y_comparaison=y_
    
          elif not parametrisation:
            # MSE: 
            MSE=((trajectoires_xy.cpu().detach()[:,0]-x)**2).mean()+((trajectoires_xy.cpu().detach()[:,1]-y)**2).mean()
            MSE+=((trajectoires_xy.cpu().detach()[:,2]-px)**2).mean()+((trajectoires_xy.cpu().detach()[:,3]-py)**2).mean()
            MSE=MSE/(4*Nt)
            # Should probably do a dict that saves them / save them to a file for the cluster
            print("The MSE for head {} is {}".format(i, MSE))

            ax[2].plot(t_comparaison.cpu().detach(),trajectoires_xy.cpu().detach()[:,0]-x)
            ax[2].set_title('Difference between NN solution and numerical solution -x ')
            ax[3].plot(t_comparaison.cpu().detach(),trajectoires_xy.cpu().detach()[:,1]-y)
            ax[3].set_title('Difference between NN solution and numerical solution - y ')

            # Compute the energy along t_comparaison
            px_comparaison=trajectoires_xy[:,2]
            py_comparaison=trajectoires_xy[:,3]

            px_comparaison=px_comparaison.reshape((-1,1))
            py_comparaison=py_comparaison.reshape((-1,1))
            x_comparaison=trajectoires_xy[:,0]
            y_comparaison=trajectoires_xy[:,1]
            x_comparaison=x_comparaison.reshape((-1,1))
            y_comparaison=y_comparaison.reshape((-1,1))

          # Theoretical energy
          print("The theoretical energy is {}".format(H0_init[i]))
          ax[4].plot(t_comparaison.cpu().detach(),H0_init[i]*np.ones(Nt), linestyle=':', c='r')
          ax[4].set_title('Energy')

          H_curr_comparaison=(px_comparaison**2+py_comparaison**2)/2
          for m in range(len(means)):
            # Get the current means
            mu_x=means[m][0]
            mu_y=means[m][1]

            # Updating the energy
            H_curr_comparaison+=-A_*torch.exp(-(1/(2*sig**2))*((x_comparaison-mu_x)**2+(y_comparaison-mu_y)**2))
          ax[4].plot(t_comparaison.cpu().detach(),H_curr_comparaison.cpu().detach())



      y1=np.linspace(-.1,1.1,500); x1= np.linspace(-.1,1.1,500)
      x, y = np.meshgrid(x1, y1)
      
      ########## Saving ###########
      # Saving the means passed in
      filename = 'Initial_x_'+str(initial_x)+'final_t_'+str(final_t)+'alpha_'+ str(alpha_)+'width_'+str(width_)+\
      'epochs_'+str(epochs_)+'grid_size_'+str(grid_size)+'Means'+'.p'
      #os.mkdir(filename)
      f=open(filename,"wb")
      pickle.dump(means_cell,f)
      f.close()

      # Saving the mesh grid
      filename = 'Initial_x_'+str(initial_x)+'final_t_'+str(final_t)+'alpha_'+ str(alpha_)+'width_'+str(width_)+\
      'epochs_'+str(epochs_)+'grid_size_'+str(grid_size)+'Grid'+'.p'
      #os.mkdir(filename)
      f=open(filename,"wb")
      pickle.dump(x,f)
      pickle.dump(y,f)
      f.close()
      #############################

      V=0
      Vx=0
      Vy=0

      
      A_=0.1
      sig=.1

      for i in means_cell:
        muX1=i[0]
        muY1=i[1]
        V+=  - A_*np.exp(- (( (x-muX1)**2 + (y-muY1)**2) / sig**2)/2) 

      ########## Saving ###########
      # Saving the values of V on the grid
      filename = 'Initial_x_'+str(initial_x)+'final_t_'+str(final_t)+'alpha_'+ str(alpha_)+'width_'+str(width_)+\
      'epochs_'+str(epochs_)+'grid_size_'+str(grid_size)+'Grid_potential_values'+'.p'
      #os.mkdir(filename)
      f=open(filename,"wb")
      pickle.dump(V,f)
      f.close()
      # files.download(filename)
      ############################# 

      ax[0].contourf(x1,y1,V, levels=20, cmap='Reds_r'); #ax[0].colorbar()
      ax[0].set_xlim(-.1,1.1)

  # Make a grid, set the title and the labels
  ax[1].legend()
  ax[1].set_title("Loss")
  ax[1].set_yscale('log')
  ax[1].set_xlabel("Number of epochs")
  ax[1].set_ylabel("Loss")

  # Make a grid, set the title and the labels
  ax[2].set_title("Errors between the solutions (NN and Numerical) - x")
  ax[2].set_xlabel('$t$')
  ax[2].set_ylabel('difference in $x$')

  # Make a grid, set the title and the labels
  ax[3].set_title("Errors between the solutions (NN and Numerical) - y")
  ax[3].set_xlabel('$t$')
  ax[3].set_ylabel('difference in $y$')

  # Make a grid, set the title and the labels
  ax[4].set_title("Energy conservation")
  ax[4].set_xlabel('$t$')
  ax[4].set_ylabel('Energy')

  t1=time.time()
  print("The elapsed time is {}".format(t1-t0))
  plt.show()

  return network2

network_base=N_heads_run_Gaussiann_transfer(random_ic=False, parametrisation=True, energy_conservation=False, max_grid_grow=400, \
                                   epochs_=25000, grid_size=400, \
                                   number_of_heads=11, number_of_heads_TL=1)

# Didn't keep any of the TL code from the colab (in Paper folder)
