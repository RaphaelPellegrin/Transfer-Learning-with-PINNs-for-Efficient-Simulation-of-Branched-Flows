# Transfer-Learning-with-PINNs-for-Efficient-Simulation-of-Branched-Flows

Code for the Paper "Transfer Learning with Physics-Informed Neural Networks for Efficient Simulation of Branched Flows", Machine Learning and the Physical Sciences Workshop at the 36th conference on Neural Information Processing Systems (NeurIPS).

https://arxiv.org/abs/2211.00214

### How to run:

create a virtual environment:

```
conda create -n <your_venv> python=3.11

conda activate <your_venv>

pip install -r requirements.txt
```

Then the base training is performed in
FFFN/Base/base_training.py

where FFNN stands for Feed-Forward Neural Network. See Blake's and Dylan's GithHubs for the GANs code.

it can be run from the terminal using:

```
python base_training.py
```

When the training of the base is finished, TL is performed by calling ```tl.py```

```
python tl.py
```

# Citing

If you use any component of this repository, please cite and reference the following:

```
Raphael Pellegrin, Blake Bullwinkel, Marios Mattheakis, and Pavlos Protopapas. Transfer
learning with physics-informed neural networks for efficient simulation of branched flows. In
NeurIPS Workshop on Machine Learning and Physical Sciences, 2022.
```



