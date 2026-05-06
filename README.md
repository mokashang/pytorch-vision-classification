# PyTorch Vision Classification Suite

Three small PyTorch classifiers covering the progression from a one-layer
logistic baseline to a regularized MLP on a harder dataset:

1. **`mnist_logistic/`** — single-layer logistic classifier on MNIST. A
   from-scratch `torch.utils.data.Dataset` that reads MNIST out of HDF5,
   SGD training with L2 weight decay, train/test log-loss and accuracy
   curves, and a normalized confusion-matrix heatmap.

2. **`fashion_mnist_regularization/`** — two MLPs on Fashion-MNIST trained
   for 40 epochs each:
   - **Model 1** — 784→128→10, no regularization.
   - **Model 2** — 784→48→10, L2 weight decay + dropout.
   The notebook plots per-layer weight histograms for each model and
   discusses how dropout + weight decay tighten the weight distribution.

3. **`cifar10_mlp/`** — a flat MLP (3072→256→128→10) on CIFAR-10 with
   dropout, L2 weight decay, and ReLU. The notebook reports the
   confusion matrix on the test split and a short discussion of why an
   MLP plateaus on CIFAR-10 (it has no spatial inductive bias —
   convolutional architectures dominate).

## Layout

```
mnist_logistic/
  mnist_logistic.ipynb
  requirements.txt
fashion_mnist_regularization/
  fashion_mnist_regularization.ipynb
  requirements.txt
cifar10_mlp/
  cifar10_mlp.ipynb
  requirements.txt
```

## Data

- **MNIST** for `mnist_logistic/` — expects `mnist_traindata.hdf5` and
  `mnist_testdata.hdf5` at the repo root (the notebook also looks in the
  current directory).
- **Fashion-MNIST** and **CIFAR-10** are pulled automatically by
  `torchvision.datasets`.

## Running

Each module is independent. Pick one and install its requirements:

```bash
pip install -r mnist_logistic/requirements.txt
jupyter notebook mnist_logistic/mnist_logistic.ipynb
```

## Stack

`torch==2.1.0`, `torchvision`, `matplotlib==3.8.0`, `h5py`, `numpy`,
`seaborn`, `scikit-learn`.
