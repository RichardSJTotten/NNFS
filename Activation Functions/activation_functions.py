import numpy as np
import matplotlib.pyplot as plot


# Linear function y=x
def linear(x):
    return x


# Step Function if x<=0 then 0 else 1
def step(x):
    return np.heaviside(x, 1)


# Sigmoid Function: 1 / (1+exp(-X))
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ReLU Function if x<=0 then 0 else X
def relu(x):
    return np.maximum(0, x)


# Create some example values for X
x = np.arange(-10, 10, 0.1)

fig, axis = plot.subplots(2, 2)
fig.set_figheight(7)
fig.set_figwidth(7)
fig.tight_layout(pad=3)

axis[0, 0].plot(x, linear(x))
axis[0, 0].set_title('Linear Function')
axis[0, 1].plot(x, step(x))
axis[0, 1].set_title('Step Function')
axis[1, 0].plot(x, sigmoid(x))
axis[1, 0].set_title('Sigmoid Function')
axis[1, 1].plot(x, relu(x))
axis[1, 1].set_title('ReLU Function')
plot.show()