# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 23:34:57 2021

@author: devanshi
"""

import numpy as np
import neurolab as nl
import random 

random.seed(1)
input_devanshi =  np.random.uniform(low= -0.6, high= 0.6, size =  20).reshape(10,2)
output_devanshi =  input_devanshi [:,0] + input_devanshi[:,1]
output_devanshi =  output_devanshi.reshape(10,1)
nn = nl.net.newff([[-0.6,0.6],[-0.6, 0.6]], [6, 1])
error_progress = nn.train(input_devanshi, output_devanshi, epochs=2000, show=15, goal=0.00001)

result1 = nn.sim([[0.1,0.2]])
