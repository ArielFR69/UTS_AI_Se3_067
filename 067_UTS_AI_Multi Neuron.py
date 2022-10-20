#Ariel Fikri Ramadhani
#21091397067
#Multiple Neuron

#Intializing Library 
import numpy as np

#Input Layer 10 Features
inputs = [1.4, 1.2, 2.7, 4.5, 3.5, 1.7, 2.0, 1.5, 0.3, 4.0]

#Neuron 5
weights = [[0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0],
		   [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
		   [0.03, 0.04, 0.97, 0.44, 0.27, 0.3, 0.35, 0.52, 0.53, 0.47],
           [0.95, 0.48, 0.07, 0.45, 0.09, 0.01, 0.55, 0.63, 0.33, 0.27],
           [0.64, 0.07, 0.09, 0.21, 0.86, 0.25, 0.07, 0.01, 0.31, 0.28]]

#Bias dari Layer

biases = [1.0, 2.0, 0.5, 2.5, 1.5]

#Calculating Layer Output

layer_outputs = np.dot(weights, inputs) + biases

#Printing Out
print(layer_outputs)