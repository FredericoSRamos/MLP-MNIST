# MNIST on Multi-Layer Perceptron

This project is a fully connected neural network implemented from scratch in pure Python, trained on the **MNIST** handwritten digits dataset.  

No machine learning libraries are used, every part of the network is manually implemented.

The goal of this project was educational, to understand how neural networks actually work.

## Features

- Fully connected feedforward neural network
- Sigmoid activation function
- Backpropagation with gradient descent
- Mean Squared Error (MSE) loss
- Custom model saving and loading using pickle
- Trained and evaluated on the MNIST dataset

## Network Architecture

- Input Layer: 784 neurons (28x28 image pixels)
- Hidden Layer 1: 128 neurons
- Hidden Layer 2: 128 neurons
- Output Layer: 10 neurons (digits 0–9)

## How It Works

1. **Forward Pass**
   - Inputs are normalized pixel values from MNIST
   - Each neuron computes a weighted sum + bias
   - Sigmoid activation is applied

2. **Loss Calculation**
   - Mean Squared Error (MSE) between predicted output and expected vector

3. **Backpropagation**
   - Errors are propagated backward layer by layer
   - Gradients are computed using the sigmoid derivative
   - Weights and biases are updated using gradient descent

4. **Model Saving**
   - The best-performing model is saved automatically

- **Note:** Saved models can be loaded directly

## Running the Project

### Requirements
- Python
- MNIST dataset in CSV format

### Steps

1. Place the MNIST CSV files inside the `datasets` directory:
   - `mnist_train.csv`
   - `mnist_test.csv`

2. Run the training and evaluation script:

   `python main.py`

After training, the model is evaluated on the test dataset and prints the accuracy

### Notes

This implementation prioritizes clarity over performance, training is relatively slow compared to optimized ML libraries. It's intended for learning purposes only.

### License
MIT License
