#Ariel Fikri Ramadhani
#2109139067
#Single Neuron

#Initialzing and Importing Numpy & Library
import numpy as np

#Layer Input 10 Features
inputs = [0.5, 0.7, 0.7, 3.4, 0.6, 0.7, 1.5, 0.3, 3.9, 1.6]

#Neuron 1
weights = [[2.7, 3.6, 3.5, 1.8, 0.0, 2.1, 0.4, 1.8, 0.5, 0.3],]

#Bias from Layer
biases = [2.0, 3.0, 1.0, 1.5, 2.5]

#Calculating Output
layer_outputs = np.dot(weights, inputs) + biases

#Printing Output
print(layer_outputs)