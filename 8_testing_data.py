from nnfs.datasets import spiral_data # Import dataset
import numpy as np
import nnfs
import matplotlib.pyplot as plt

"""
nnfs.init():
	Sets the random seed to 0
	Creates a float32 dtype default
	Overrides original dot product 
"""
nnfs.init()

X, y = spiral_data(samples=100, classes=3)

plt.scatter(X[:, 0], X[:, 1])
plt.show()

