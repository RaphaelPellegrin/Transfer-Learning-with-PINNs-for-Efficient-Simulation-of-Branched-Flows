# Transfer-Learning-with-PINNs-for-Efficient-Simulation-of-Branched-Flows

Code for the Paper "Transfer Learning with Physics-Informed Neural Networks for Efficient Simulation of Branched Flows", Machine Learning and the Physical Sciences Workshop at the 36th conference on Neural Information Processing Systems (NeurIPS).

https://arxiv.org/abs/2211.00214

### How to run:

create a virtual environment:

```
conda create -n <your_venv> python=3.11

conda activate <your_venv>

pip install -e .
```

Then the base training is performed in ```scripts/base_training.py```

it can be run from the terminal using:

```
python scripts/base_training.py
```

When the training of the base is finished, TL is performed by calling ```scripts/tl.py```

```
python scripts/tl.py
```

# Citing

If you use any component of this repository, please cite and reference the following:

```
Raphael Pellegrin, Blake Bullwinkel, Marios Mattheakis, and Pavlos Protopapas. Transfer
learning with physics-informed neural networks for efficient simulation of branched flows. In
NeurIPS Workshop on Machine Learning and Physical Sciences, 2022.
```



