# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 23:53:59 2021

@author: devanshi
"""


import numpy as np
import matplotlib.pyplot as plt
import neurolab as nl
import os

np.random.seed = 1

x1 = np.random.uniform(size=10, low=-0.6, high=0.6)
x2 = np.random.uniform(size=10, low=-0.6, high=0.6)
x3 = np.random.uniform(size=10, low=-0.6, high=0.6)

y = x1 + x2 + x3

input_devanshi = np.stack((x1, x2, x3), axis=1)
labels = y[:]
output_devanshi = labels.reshape((labels.shape[0], 1))



# Minimum and maximum values for each dimension


# Define a single-layer neural network

nn = nl.net.newff([[-0.6,0.6],[-0.6,0.6],[-0.6,0.6]], [6,1])

# Train the neural network
error_progress = nn.train(input_devanshi, output_devanshi, epochs=100, show=15, goal=0.00001)

# Run the classifier on test datapoints
result_05 = nn.sim([[0.2,0.1,0.2]])
print("Result is05 ", result_05)
nn2 = nl.net.newff([[-0.6,0.6],[-0.6,0.6],[-0.6,0.6]], [5,3,1])


# Set the training algorithm to gradient descent
nn2.trainf = nl.train.train_gd
# Train the neural network
error_progress2 = nn2.train(input_devanshi, output_devanshi, epochs=1000, show=100, goal=0.00001)
# Plot training error
plt.figure() 
plt.plot(error_progress2)
plt.xlabel('Number of epochs')
plt.ylabel('Error')
plt.title('Training error progress')
# Run the classifier on test datapoints
result_06 = nn.sim([[0.2,0.1,0.2]])
print("Result06 is ", result_06)
