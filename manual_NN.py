import numpy as np

# Sigmoid activation function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Initialize weights & biases
def initialize_parameters(input_size, hidden_size, output_size):
    np.random.seed(0)
    W1 = np.random.randn(hidden_size, input_size) * 0.01
    b1 = np.zeros((hidden_size, 1))
    W2 = np.random.randn(output_size, hidden_size) * 0.01
    b2 = np.zeros((output_size, 1))
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

# Forward propagation
def forward_propagation(X, parameters):
    W1, b1, W2, b2 = parameters["W1"], parameters["b1"], parameters["W2"], parameters["b2"]
    
    Z1 = np.dot(W1, X) + b1
    A1 = sigmoid(Z1)
    
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    
    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
    return A2, cache

# Compute loss (Binary Cross-Entropy)
def compute_loss(Y, A2):
    m = Y.shape[1]
    loss = (-1/m) * np.sum(Y * np.log(A2) + (1 - Y) * np.log(1 - A2))
    return loss

# Backward propagation (Gradient computation)
def backward_propagation(X, Y, parameters, cache):
    m = X.shape[1]
    A1, A2 = cache["A1"], cache["A2"]
    W2 = parameters["W2"]
    
    dZ2 = A2 - Y
    dW2 = (1/m) * np.dot(dZ2, A1.T)
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
    
    dZ1 = np.dot(W2.T, dZ2) * (A1 * (1 - A1))
    dW1 = (1/m) * np.dot(dZ1, X.T)
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)
    
    gradients = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
    return gradients

# Update parameters using gradient descent
def update_parameters(parameters, gradients, learning_rate):
    parameters["W1"] -= learning_rate * gradients["dW1"]
    parameters["b1"] -= learning_rate * gradients["db1"]
    parameters["W2"] -= learning_rate * gradients["dW2"]
    parameters["b2"] -= learning_rate * gradients["db2"]
    return parameters

# Training setup
learning_rate = 0.01
epochs = 1000
parameters = initialize_parameters(input_size=2, hidden_size=4, output_size=1)

# Generate synthetic dataset
np.random.seed(0)
X = np.random.randn(2, 100) * 10  # Shape (features, samples)
Y = (X[0, :] + X[1, :] > 10).astype(int).reshape(1, -1)  # Shape (1, samples)

# Training loop
for i in range(epochs):
    A2, cache = forward_propagation(X, parameters)
    loss = compute_loss(Y, A2)
    gradients = backward_propagation(X, Y, parameters, cache)
    parameters = update_parameters(parameters, gradients, learning_rate)

    if i % 100 == 0:  # Print loss every 100 epochs
        print(f"Epoch {i}: Loss = {loss:.4f}")

import matplotlib.pyplot as plt
losses = []  # Store loss values
for i in range(epochs):
    A2, cache = forward_propagation(X, parameters)
    loss = compute_loss(Y, A2)
    losses.append(loss)  # Save loss
    gradients = backward_propagation(X, Y, parameters, cache)
    parameters = update_parameters(parameters, gradients, learning_rate)
plt.plot(range(epochs), losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.show()