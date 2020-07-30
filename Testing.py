# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 15:51:19 2020

@author: Dell
"""


import numpy as np
from Perceptron1 import Perceptron

training_inputs = []
training_inputs.append(np.array([1,1]))
training_inputs.append(np.array([1,0]))
training_inputs.append(np.array([0,1]))
training_inputs.append(np.array([0,0]))


labels = np.array([1,0,0,0])

Perceptron1 = Perceptron(2)
Perceptron1.train(training_inputs,labels)


inputs = np.array([1,1])
print(Perceptron1.predict(inputs))

print("========")

inputs = np.array([0,1])
print(Perceptron1.predict(inputs))


print("final weights --> " + Perceptron1.weights)
