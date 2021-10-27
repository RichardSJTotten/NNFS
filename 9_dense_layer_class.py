import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

# Dense Layer
class Layer_Dense:
	
	# Layer initialisation
	def __init__(self, n_inputs, n_neurons):
		# Initialise weights and biases
		self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
		self.biases = np.zeros((1, n_neurons))
	
	# Forward Pass
	def forward(self, inputs):
		# Calculate output values based on inputs, weights and biases
		self.output = np.dot(inputs, self.weights) + self.biases
    
#ReLU Activation Function
class Activation_ReLU:
    # Forward Pass
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

# Create dataset
X, y = spiral_data(samples=100, classes=3)

# Create dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2,3)

# Create activation
activation_1 = Activation_ReLU()

# Perform a forward pass of our training data through the layer
dense1.forward(X)

#Perform activation
activation_1.forward(dense1.output)

# print(dense1.output[:5])
print(activation_1.output[:5])
