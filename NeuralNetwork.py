# -*- coding: utf-8 -*-
import numpy as np
import scipy.special 
import matplotlib.pyplot as plt

class NeuralNetwork:
    
    def __init__(self, inputNodes, hiddenNodes, outputNodes, learningrate):
        self.inputNodes = inputNodes
        self.hiddenNodes = hiddenNodes
        self.outputNodes = outputNodes
        self.weightInputToHidden = np.random.normal(0.0, pow(self.inputNodes, -0.5), (self.hiddenNodes, self.inputNodes))
        self.weightHiddenToOutput = np.random.normal(0.0, pow(self.hiddenNodes, -0.5), (self.outputNodes, self.hiddenNodes))
        self.learningrate = learningrate
        # activation function is the sigmoid function
        self.activation_function = scipy.special.expit       

    def train(self, inputs_list, targets_list):
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        
        hidden_inputs = np.dot(self.weightInputToHidden, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        final_inputs = np.dot(self.weightHiddenToOutput, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        
        output_errors = targets - final_outputs
        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = np.dot(self.weightHiddenToOutput.T, output_errors) 
        
        # update the weights for the links between the hidden and output layers
        self.weightHiddenToOutput += self.learningrate * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))
        
        # update the weights for the links between the input and hidden layers
        self.weightInputToHidden += self.learningrate * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))
    

    def query(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T
        
        hidden_inputs = np.dot(self.weightInputToHidden, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        final_inputs = np.dot(self.weightHiddenToOutput, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs


if __name__=="__main__":
    testNetwork = NeuralNetwork(3, 3, 3, 0.5)
    print(testNetwork.query([1.0, 0.5, -1.5]))
    trainData = 0
    with open("mnist_train_100.csv") as f:
        trainData = f.readlines()
    print(len(trainData))
    
    all_values = trainData[0].split(',') 
    image_array = np.asfarray(all_values[1:]).reshape((28,28)) 
    plt.imshow(image_array, cmap='Greys', interpolation='None')
    
    