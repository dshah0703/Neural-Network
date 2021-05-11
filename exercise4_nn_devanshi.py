# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 23:48:35 2021

@author: devanshi
"""

import numpy as np
import neurolab as nl
import random 
import matplotlib.pyplot as plt

""" to generate the same number"""
random.seed(1)
input_devanshi =  np.random.uniform(low= -0.6, high= 0.6, size =  200).reshape(100,2)
output_devanshi =  input_devanshi[:,0] + input_devanshi[:,1]
output_devanshi =  output_devanshi.reshape(100,1)
nn = nl.net.newff([[-0.6,0.6],[-0.6, 0.6]], [5,3, 1])
nn.trainf = nl.train.train_gd
"""the epoch will start from number 100 and the distance between two numbers will be 100 and end up on 1000"""
error_progress = nn.train(input_devanshi, output_devanshi, epochs=1000, show=100, goal=0.00001)
plt.figure() 
plt.plot(error_progress)
plt.xlabel('Number of epochs')
plt.ylabel('Error')
plt.title('Training error progress')
temp = np.array([[0.1,0.2]])
result3 = nn.sim(temp)
print(result3)