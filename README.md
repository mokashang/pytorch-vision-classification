# EE541 Homework 8

**Name:** Mengjia Shang
**Student ID:** 7338151449

## Description

This submission implements three deep learning classification tasks using PyTorch.

---

## q1 — Logistic Classifier on MNIST

Files:
- `q1.ipynb` — custom HDF5 dataset, single-layer logistic classifier, SGD training with L2 regularization, learning curves, confusion matrix
- `requirements.txt`
- `learning_curves.pdf` — train/test log-loss and accuracy vs. epochs
- `confusion_matrix.pdf` — normalized confusion matrix heatmap

Data files (at repo root): `mnist_traindata.hdf5`, `mnist_testdata.hdf5`

---

## q2 — Regularization and Dropout on Fashion MNIST

Files:
- `q2.ipynb` — two MLP models trained for 40 epochs, weight histograms, analysis of weight distributions
- `requirements.txt`
- `model1_weights.pdf` — histograms for Model 1 (128 nodes, no regularization)
- `model2_weights.pdf` — histograms for Model 2 (48 nodes, L2 + dropout)

---

## q3 — CIFAR-10 Classification

Files:
- `q3.ipynb` — MLP (3072→256→128→10) with dropout and L2 regularization, confusion matrix, answers to questions
- `requirements.txt`
- `confusion_matrix.pdf` — normalized confusion matrix heatmap

---

## Dependencies

See `requirements.txt` in each problem directory. Main dependencies:
- `torch==2.1.0`
- `torchvision`
- `matplotlib==3.8.0`
- `h5py`, `numpy`, `seaborn`, `scikit-learn`
