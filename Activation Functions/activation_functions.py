import numpy as np
import matplotlib.pyplot as plot

# Linear function y=x
def linear(X):
    return X

# Step Function if x<=0 then 0 else 1
def step(X):
    return np.heaviside(X,1)

# Sigmoid Function: 1 / (1+exp(-X))
def sigmoid(X):
    return 1 / (1+np.exp(-X))

# ReLU Function if x<=0 then 0 else X
def relu(X):
    return np.maximum(0,X)

# Create some example values for X
X = np.arange(-10, 10, 0.1)

fig, axis = plot.subplots(2,2)
fig.set_figheight(7)
fig.set_figwidth(7)
fig.tight_layout(pad=3)

axis[0,0].plot(X, linear(X))
axis[0,0].set_title('Linear Function')
axis[0,1].plot(X, step(X))
axis[0,1].set_title('Step Function')
axis[1,0].plot(X, sigmoid(X))
axis[1,0].set_title('Sigmoid Function')
axis[1,1].plot(X, relu(X))
axis[1,1].set_title('ReLU Function')
