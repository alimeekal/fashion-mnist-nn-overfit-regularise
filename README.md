# Building, Breaking and Fixing a Neural Network

A feedforward neural network built end-to-end on Fashion-MNIST — implemented from scratch in NumPy, verified against PyTorch, deliberately pushed into overfitting, then repaired using regularisation and hyperparameter tuning.

## Overview

| Part | Description |
|------|-------------|
| 1 | Backpropagation implemented from scratch in NumPy, verified against PyTorch autograd |
| 2 | Baseline model and activation function study (sigmoid, tanh, ReLU, leaky ReLU) |
| 3 | Loss function comparison (cross-entropy vs. MSE) |
| 4 | Optimiser comparison (SGD, SGD+momentum, RMSprop, Adam) |
| 5 | Deliberate overfitting on a reduced training set |
| 6 | Regularisation study (L2, L1, dropout, batch norm, early stopping, data augmentation, more data) |
| 7 | Hyperparameter tuning via random search + 5-fold cross-validation, final test evaluation |

## Requirements

- Kaggle notebook with GPU T4 x2 accelerator
- Python packages: `numpy`, `pandas`, `scikit-learn`, `torch`, `matplotlib`, `seaborn`

## Dataset

Fashion-MNIST: https://www.kaggle.com/datasets/zalando-research/fashionmnist

Add it via **Add Input** → search "Fashion MNIST" → select `zalando-research/fashionmnist`.

## Reproducing the Results

1. Open the notebook on Kaggle with GPU T4 x2 enabled.
2. Add the Fashion-MNIST dataset as an input.
3. Run all cells top to bottom (**Run All**) — later parts depend on models/variables from earlier parts.
4. Random seed is fixed at `SEED = 33` throughout (NumPy, PyTorch, train/val splits, k-fold), so results are reproducible on re-run.

## Key Results

- Gradient check (Part 1): max absolute difference vs. PyTorch on the order of 1e-8
- Forced overfitting (Part 5): 100% training accuracy vs. 82.4% validation accuracy
- Best regulariser (Part 6): increasing training data to 20,000 samples, reducing the generalisation gap from 0.176 to 0.071
- Final tuned model (Part 7): 83.19% test accuracy, macro F1 of 0.830

## Repository Structure

```
.
├── README.md
└── notebook.ipynb
```
