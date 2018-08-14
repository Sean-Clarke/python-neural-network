import numpy
import matplotlib

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
                self.layers[l][n]['value'] = influence / load
        
    def backpropogate(self, answer):
        """Adjusts weights based on comparison between output and answer"""
        
    def train(self):
        """Trains this network with given dataset/answer key, adjusting/optimizing weights"""
        pass
            
    def test(self, data):
        """Tests this network on given dataset/answer key"""
        pass
    
