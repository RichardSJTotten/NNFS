import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()


# Dense Layer
class LayerDense:

    # Layer initialisation
    def __init__(self, n_inputs, n_neurons):
        # Initialise weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    # Forward Pass
    def forward(self, inputs):
        # Calculate output values based on inputs, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases


# ReLU Activation Function
class ActivationReLU:
    # Forward Pass
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


# Softmax Activation Function
class ActivationSoftmax:
    # Forward Pass
    def forward(self, inputs):
        # Get normalised probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        # Normalise them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities


# Create dataset
X, y = spiral_data(samples=100, classes=3)

# Create dense layer with 2 input features and 3 output values
dense1 = LayerDense(2, 3)

# Create activation
activation_1 = ActivationReLU()

# Create 2nd dense layer (3 inputs to match first layers outputs)
dense2 = LayerDense(3, 3)

# Create Softmax activation
activation_2 = ActivationSoftmax()

# Perform a forward pass of our training data through the layer
dense1.forward(X)

# Perform activation
activation_1.forward(dense1.output)

# Perform forward pass through layer 2
dense2.forward(activation_1.output)

# Forward pass through 2nd activation
activation_2.forward(dense2.output)

#  print(dense1.output[:5])
print(activation_2.output[:5])
