import numpy
import matplotlib
import random

def sigmoid(v):
    return(numpy.exp(v) / (1 + numpy.exp(v)))
    
def sigmoid_derivative(v):
    return sigmoid(v)*(1 - sigmoid(v))

def softmax(values):
    return [numpy.exp(v)/numpy.sum(numpy.exp(values)) for v in values]

class Network:
    """Represents a neural network"""
    def __init__(self, architecture):
        """Class initialization special method, architecture is supplied as follows: [inputs, *width_of_hidden_layer_0...*width_of_hidden_layer_n, outputs]"""
        self.depth = len(architecture) - 1
        self.layers = [[]]
        for i in range(architecture[0]):
            self.layers[0].append({'node':'i' + str(i), 'value':None})
        for l in range(1, self.depth):
            self.layers.append([])
            for h in range(architecture[l]):
                weights = []
                for n in range(architecture[l-1]):
                    weights.append(None)
                self.layers[l].append({'weights':weights, 'value':None})
        self.layers.append([])
        for o in range(architecture[-1]):
            weights = []
            for n in range(architecture[-2]):
                weights.append(None)
            self.layers[-1].append({'node':'o' + str(o), 'weights':weights, 'value':None})
            
    def __repr__(self):
        """Class representation special method"""
        architecture = []
        for l in self.layers:
            architecture.append(len(l))
        return str(architecture)
                
    def add_layer(self, width, list_of_weights, index=-1):
        """Adds a new hidden layer with width nodes to the network, in given or default -1 index, with weights mapped from list_of_weights"""
        if index >= len(self.layers):
            raise ValueError('You cannot add a new hidden layer after the output layer')
        if index == 0 or index + len(self.layers) <= 0:
            index = 1
        self.depth += 1
        if list_of_weights == []:
            for n in range(width):
                weights = []
                for i in self.layers[index - 1]:
                    weights.append(None)
                list_of_weights.append(weights)
        for n in range(len(self.layers[index])):
            weights = []
            for i in range(width):
                weights.append(None)
            self.layers[index][n]['weights'] = weights
        nl = []
        for n in range(width):
            nl.append({'weights':[list_of_weights[n]], 'value':None})
        self.layers.insert(index, nl)

    def randomize(self):
        """Randomizes the networks weights"""
        for l in range(1, len(self.layers)):
            for n in range(len(self.layers[l])):
                for w in range(len(self.layers[l][n]['weights'])):
                    self.layers[l][n]['weights'][w] = numpy.random.rand()
        
    def analyze(self, input):
        """Analzyes given list or matrix of input data, returning network outputs"""
        if len(input) != len(self.layers[0]):
            raise ValueError('number of values in input must match the network\'s input layer width (ie. number of nodes in Network.layers[0])')
        for i in range(len(input)):
            self.layers[0][i]['value'] = input[i]
        for l in range(1, len(self.layers)):
            for n in range(len(self.layers[l])):
                influence = 0
                load = 0
                for w in range(len(self.layers[l][n]['weights'])):
                    influence += (self.layers[l][n]['weights'][w] * self.layers[l-1][w]['value'])
                    load += self.layers[l][n]['weights'][w]
                self.layers[l][n]['value'] = sigmoid(influence / load)
        if len(self.layers[-1]) == 1:
            output = self.layers[-1][0]['value']
        else:
            output = softmax([self.layers[-1][o]['value'] for o in range(len(self.layers[-1])))
        return output
        
    def backpropogate(self, output, answer, learning_rate):
        """Adjusts weights based on comparison between output and answer"""
        loss = 0.5 * (answer - output)**2
        for l in range(len(self.layers[1:])):
            for n in range(len(self.layers[l])):
                for w in range(len(self.layers[l][n]['weights'])):
                    delta = learning_rate * (answer - output) * #[self.layers[l][n]['weights'][i]] for i in range(len(self.layers[l][n]['weights']))]) * 
                    self.layers[l][n]['weights'][w] += delta
        
    def train(self, data):
        """Trains this network with given dataset/answer key, adjusting/optimizing weights"""
        pass
            
    def test(self, data):
        """Tests this network on given dataset/answer key"""
        for d in data:
            pass
        
# for testing
iris_dataset = [[5.1, 3.5, 1.4, 0.2, 'I. setosa'],[4.9, 3.0, 1.4, 0.2, 'I. setosa'],[4.7, 3.2, 1.3, 0.2, 'I. setosa'],[4.6, 3.1, 1.5, 0.2, 'I. setosa'],[5.0, 3.6, 1.4, 0.3, 'I. setosa'],[5.4, 3.9, 1.7, 0.4, 'I. setosa'],[4.6, 3.4, 1.4, 0.3, 'I. setosa'],[5.0, 3.4, 1.5, 0.2, 'I. setosa'],[4.4, 2.9, 1.4, 0.2, 'I. setosa'],[4.9, 3.1, 1.5, 0.1, 'I. setosa'],[5.4, 3.7, 1.5, 0.2, 'I. setosa'],[4.8, 3.4, 1.6, 0.2, 'I. setosa'],[4.8, 3.0, 1.4, 0.1, 'I. setosa'],[4.3, 3.0, 1.1, 0.1, 'I. setosa'],[5.8, 4.0, 1.2, 0.2, 'I. setosa'],[5.7, 4.4, 1.5, 0.4, 'I. setosa'],[5.4, 3.9, 1.3, 0.4, 'I. setosa'],[5.1, 3.5, 1.4, 0.3, 'I. setosa'],[5.7, 3.8, 1.7, 0.3, 'I. setosa'],[5.1, 3.8, 1.5, 0.3, 'I. setosa'],[5.4, 3.4, 1.7, 0.2, 'I. setosa'],[5.1, 3.7, 1.5, 0.4, 'I. setosa'],[4.6, 3.6, 1.0, 0.2, 'I. setosa'],[5.1, 3.3, 1.7, 0.5, 'I. setosa'],[4.8, 3.4, 1.9, 0.2, 'I. setosa'],[5.0, 3.0, 1.6, 0.2, 'I. setosa'],[5.0, 3.4, 1.6, 0.4, 'I. setosa'],[5.2, 3.5, 1.5, 0.2, 'I. setosa'],[5.2, 3.4, 1.4, 0.2, 'I. setosa'],[4.7, 3.2, 1.6, 0.2, 'I. setosa'],[4.8, 3.1, 1.6, 0.2, 'I. setosa'],[5.4, 3.4, 1.5, 0.4, 'I. setosa'],[5.2, 4.1, 1.5, 0.1, 'I. setosa'],[5.5, 4.2, 1.4, 0.2, 'I. setosa'],[4.9, 3.1, 1.5, 0.2, 'I. setosa'],[5.0, 3.2, 1.2, 0.2, 'I. setosa'],[5.5, 3.5, 1.3, 0.2, 'I. setosa'],[4.9, 3.6, 1.4, 0.1, 'I. setosa'],[4.4, 3.0, 1.3, 0.2, 'I. setosa'],[5.1, 3.4, 1.5, 0.2, 'I. setosa'],[5.0, 3.5, 1.3, 0.3, 'I. setosa'],[4.5, 2.3, 1.3, 0.3, 'I. setosa'],[4.4, 3.2, 1.3, 0.2, 'I. setosa'],[5.0, 3.5, 1.6, 0.6, 'I. setosa'],[5.1, 3.8, 1.9, 0.4, 'I. setosa'],[4.8, 3.0, 1.4, 0.3, 'I. setosa'],[5.1, 3.8, 1.6, 0.2, 'I. setosa'],[4.6, 3.2, 1.4, 0.2, 'I. setosa'],[5.3, 3.7, 1.5, 0.2, 'I. setosa'],[5.0, 3.3, 1.4, 0.2, 'I. setosa'],[7.0, 3.2, 4.7, 1.4, 'I. versicolor'],[6.4, 3.2, 4.5, 1.5, 'I. versicolor'],[6.9, 3.1, 4.9, 1.5, 'I. versicolor'],[5.5, 2.3, 4.0, 1.3, 'I. versicolor'],[6.5, 2.8, 4.6, 1.5, 'I. versicolor'],[5.7, 2.8, 4.5, 1.3, 'I. versicolor'],[6.3, 3.3, 4.7, 1.6, 'I. versicolor'],[4.9, 2.4, 3.3, 1.0, 'I. versicolor'],[6.6, 2.9, 4.6, 1.3, 'I. versicolor'],[5.2, 2.7, 3.9, 1.4, 'I. versicolor'],[5.0, 2.0, 3.5, 1.0, 'I. versicolor'],[5.9, 3.0, 4.2, 1.5, 'I. versicolor'],[6.0, 2.2, 4.0, 1.0, 'I. versicolor'],[6.1, 2.9, 4.7, 1.4, 'I. versicolor'],[5.6, 2.9, 3.6, 1.3, 'I. versicolor'],[6.7, 3.1, 4.4, 1.4, 'I. versicolor'],[5.6, 3.0, 4.5, 1.5, 'I. versicolor'],[5.8, 2.7, 4.1, 1.0, 'I. versicolor'],[6.2, 2.2, 4.5, 1.5, 'I. versicolor'],[5.6, 2.5, 3.9, 1.1, 'I. versicolor'],[5.9, 3.2, 4.8, 1.8, 'I. versicolor'],[6.1, 2.8, 4.0, 1.3, 'I. versicolor'],[6.3, 2.5, 4.9, 1.5, 'I. versicolor'],[6.1, 2.8, 4.7, 1.2, 'I. versicolor'],[6.4, 2.9, 4.3, 1.3, 'I. versicolor'],[6.6, 3.0, 4.4, 1.4, 'I. versicolor'],[6.8, 2.8, 4.8, 1.4, 'I. versicolor'],[6.7, 3.0, 5.0, 1.7, 'I. versicolor'],[6.0, 2.9, 4.5, 1.5, 'I. versicolor'],[5.7, 2.6, 3.5, 1.0, 'I. versicolor'],[5.5, 2.4, 3.8, 1.1, 'I. versicolor'],[5.5, 2.4, 3.7, 1.0, 'I. versicolor'],[5.8, 2.7, 3.9, 1.2, 'I. versicolor'],[6.0, 2.7, 5.1, 1.6, 'I. versicolor'],[5.4, 3.0, 4.5, 1.5, 'I. versicolor'],[6.0, 3.4, 4.5, 1.6, 'I. versicolor'],[6.7, 3.1, 4.7, 1.5, 'I. versicolor'],[6.3, 2.3, 4.4, 1.3, 'I. versicolor'],[5.6, 3.0, 4.1, 1.3, 'I. versicolor'],[5.5, 2.5, 4.0, 1.3, 'I. versicolor'],[5.5, 2.6, 4.4, 1.2, 'I. versicolor'],[6.1, 3.0, 4.6, 1.4, 'I. versicolor'],[5.8, 2.6, 4.0, 1.2, 'I. versicolor'],[5.0, 2.3, 3.3, 1.0, 'I. versicolor'],[5.6, 2.7, 4.2, 1.3, 'I. versicolor'],[5.7, 3.0, 4.2, 1.2, 'I. versicolor'],[5.7, 2.9, 4.2, 1.3, 'I. versicolor'],[6.2, 2.9, 4.3, 1.3, 'I. versicolor'],[5.1, 2.5, 3.0, 1.1, 'I. versicolor'],[5.7, 2.8, 4.1, 1.3, 'I. versicolor'],[6.3, 3.3, 6.0, 2.5, 'I. virginica'],[5.8, 2.7, 5.1, 1.9, 'I. virginica'],[7.1, 3.0, 5.9, 2.1, 'I. virginica'],[6.3, 2.9, 5.6, 1.8, 'I. virginica'],[6.5, 3.0, 5.8, 2.2, 'I. virginica'],[7.6, 3.0, 6.6, 2.1, 'I. virginica'],[4.9, 2.5, 4.5, 1.7, 'I. virginica'],[7.3, 2.9, 6.3, 1.8, 'I. virginica'],[6.7, 2.5, 5.8, 1.8, 'I. virginica'],[7.2, 3.6, 6.1, 2.5, 'I. virginica'],[6.5, 3.2, 5.1, 2.0, 'I. virginica'],[6.4, 2.7, 5.3, 1.9, 'I. virginica'],[6.8, 3.0, 5.5, 2.1, 'I. virginica'],[5.7, 2.5, 5.0, 2.0, 'I. virginica'],[5.8, 2.8, 5.1, 2.4, 'I. virginica'],[6.4, 3.2, 5.3, 2.3, 'I. virginica'],[6.5, 3.0, 5.5, 1.8, 'I. virginica'],[7.7, 3.8, 6.7, 2.2, 'I. virginica'],[7.7, 2.6, 6.9, 2.3, 'I. virginica'],[6.0, 2.2, 5.0, 1.5, 'I. virginica'],[6.9, 3.2, 5.7, 2.3, 'I. virginica'],[5.6, 2.8, 4.9, 2.0, 'I. virginica'],[7.7, 2.8, 6.7, 2.0, 'I. virginica'],[6.3, 2.7, 4.9, 1.8, 'I. virginica'],[6.7, 3.3, 5.7, 2.1, 'I. virginica'],[7.2, 3.2, 6.0, 1.8, 'I. virginica'],[6.2, 2.8, 4.8, 1.8, 'I. virginica'],[6.1, 3.0, 4.9, 1.8, 'I. virginica'],[6.4, 2.8, 5.6, 2.1, 'I. virginica'],[7.2, 3.0, 5.8, 1.6, 'I. virginica'],[7.4, 2.8, 6.1, 1.9, 'I. virginica'],[7.9, 3.8, 6.4, 2.0, 'I. virginica'],[6.4, 2.8, 5.6, 2.2, 'I. virginica'],[6.3, 2.8, 5.1, 1.5, 'I. virginica'],[6.1, 2.6, 5.6, 1.4, 'I. virginica'],[7.7, 3.0, 6.1, 2.3, 'I. virginica'],[6.3, 3.4, 5.6, 2.4, 'I. virginica'],[6.4, 3.1, 5.5, 1.8, 'I. virginica'],[6.0, 3.0, 4.8, 1.8, 'I. virginica'],[6.9, 3.1, 5.4, 2.1, 'I. virginica'],[6.7, 3.1, 5.6, 2.4, 'I. virginica'],[6.9, 3.1, 5.1, 2.3, 'I. virginica'],[5.8, 2.7, 5.1, 1.9, 'I. virginica'],[6.8, 3.2, 5.9, 2.3, 'I. virginica'],[6.7, 3.3, 5.7, 2.5, 'I. virginica'],[6.7, 3.0, 5.2, 2.3, 'I. virginica'],[6.3, 2.5, 5.0, 1.9, 'I. virginica'],[6.5, 3.0, 5.2, 2.0, 'I. virginica'],[6.2, 3.4, 5.4, 2.3, 'I. virginica'],[5.9, 3.0, 5.1, 1.8, 'I. virginica']]
random.shuffle(iris_dataset)
iris_training_set = iris_dataset[:int(len(iris_dataset) * 0.8)]
iris_test_set = iris_dataset[int(len(iris_dataset) * 0.8):]
