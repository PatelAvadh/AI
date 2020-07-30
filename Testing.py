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

inputs = np.array([0,1])
print(Perceptron1.predict(inputs))

print("final weights --> " + Perceptron1.weights)

'''
1) for learning rate 0.01 and thresold =5
--------------------
Epoch number :0
weights --->  [0. 0. 0.]
--------------------
Epoch number :1
weights --->  [-0.01  0.    0.  ]
--------------------
Epoch number :2
weights --->  [-0.01  0.    0.01]
--------------------
Epoch number :3
weights --->  [-0.02  0.    0.01]
--------------------
Epoch number :4
weights --->  [-0.02  0.01  0.01]
--------------------
Epoch number :5
weights --->  [-0.02  0.01  0.02]
--------------------
Epoch number :6
weights --->  [-0.02  0.01  0.02]
1
---------------------------
0
final weights --> [-0.02  0.01  0.02]
'''
