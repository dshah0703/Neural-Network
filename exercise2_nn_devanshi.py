# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 23:43:34 2021

@author: devanshi
"""

import numpy as np
import neurolab as nl
import random 
random.seed(1)
input_devanshi =  np.random.uniform(low= -0.6, high= 0.6, size =  20).reshape(10,2)
output_devanshi =  input_devanshi[:,0] + input_devanshi[:,1]
output_devanshi =  output_devanshi.reshape(10,1)


nn = nl.net.newff([[-0.6,0.6],[-0.6, 0.6]], [5,3,1])
nn.trainf = nl.train.train_gd
error_progress = nn.train(input_devanshi, output_devanshi, epochs=1000, show=100, goal=0.00001)
temp = np.array([[0.1,0.2]])
result2 = nn.sim(temp)
print(result2)
