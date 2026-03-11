# MNIST Classification Modular Solver

A modular PyTorch and Scikit-Learn implementation exploring the trade-offs between dimensionality reduction, model complexity, and spatial feature extraction on the MNIST dataset.

---

## Overview
This project serves as a technical demonstration of how **Feature Engineering** and **Architectural Complexity** influence model performance. By decoupling the data pipeline from the model architecture, this allows for experimentation between traditional statistical methods and modern Deep Learning.

---

## Features

### 1. Dynamic Feature Selection
* **Raw Pixels:** Direct input of 784 features ($28 \times 28$ flattened).
* **PCA (Principal Component Analysis):** Unsupervised dimensionality reduction used to preserve maximum variance with less dimensions.

### 2. Scalable Model Architectures
* **Simple Linear:** A single-layer baseline to establish a performance floor.
* **Multi-Layer Perceptron (MLP):** A fully connected neural network with hidden layers and **ReLU** activation.
* **Dropout Regularization:** Integration of Stochastic Dropout to mitigate overfitting by preventing co-adaptation of neurons.
* **CNN (Convolutional Neural Network):** A 2D spatial feature extractor utilizing `Conv2d` and `MaxPool2d` layers—designed to achieve state-of-the-art accuracy on image data.

---

## Modular Architecture
| File | Responsibility |
| :--- | :--- |
| **`run.py`** | **The Orchestrator:** Handles user inputs, the training loop, and evaluation metrics. |
| **`features.py`** | **The Data Pipeline:** Manages flattening, NumPy/Tensor conversions, and Scikit-Learn PCA transformations. |
| **`models.py`** | **The Factory:** Contains the PyTorch `nn.Module` classes and logic for switching between architectures. |

---