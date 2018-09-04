import random

class Neuron:
    """Represents a single node in a simple feed-forward neural network supporting backpropagation"""
    def __init__(self):
        self.input = None
        self.activation = None
        self.derivative = None
        self.error = None
        self.effect = None
        
    def __repr__(self):
        return str(self.activation)
        
    def get_input(self):
        return self.input()
        
    def get_activation(self):
        return self.activation
        
    def get_derivative(self):
        return self.derivative
        
    def get_error(self):
        return self.error
        
    def get_effect(self):
        return self.effect
        
    def activate(self):
        self.activation = 1.0 / (1.0 + 2.718281828459**-self.input)
        
    def derive(self):
        self.derivative = self.activation * (1.0 - self.activation)
        
    def evaluate(self, actual):
        self.error = 0.5 * (actual - self.activation)**2
        self.effect = self.activation - actual
        
    def set_input(self, input):
        self.input = input
        self.activate()
        self.derive()
        
    def set_activation(self, activation):
        self.activation = activation
        self.derive()
        
    def set_derivative(self, derivative):
        self.derivative = derivative
        
    def set_effect(self, effect):
        self.effect = effect
        

class Network:
    """Represents a simple feed-forward neural network capable of weight optimization via backpropagation"""
    def __init__(self, architecture):
        self.layers = [[Neuron() for n in range(l)] for l in architecture]
        self.weights = [[[random.random() for w in range(len(self.layers[l - 1]))] for h in range(len(self.layers[l]))] for l in range(1, len(self.layers))]
        self.biases = False
    def __repr__(self):
        rstr = ''
        for l in range(len(self.layers)):
            for n in self.layers[l]:
                rstr += str(n.get_activation()) + '  '
            rstr += '\n'
        return rstr
        
     
    def print_inputs(self):
        print(str(self.layers[0]))
        
    def print_outputs(self):
        print(str(self.layers[-1]))
     
    def print_nodes(self):
        print('\n'.join([str(layer) for layer in self.layers]))
        
    def print_weights(self):
        print('\n'.join([str(weight) for weight in self.weights]))
        
    def set_inputs(self, inputs):
        if len(inputs) != len(self.layers[0]):
            raise ValueError("Length of given input array must equal the called network's input layer width!")
        for i in range(len(inputs)):
            self.layers[0][i].set_activation(inputs[i])
            
    def set_weights(self, weights):
        self.weights = weights
        
    def set_biases(self, biases):
        self.biases = biases
    
    def feedforward(self):
        for l in range(1, len(self.layers)):
            for n in range(len(self.layers[l])):
                net = sum([self.weights[l - 1][n][p] * self.layers[l - 1][p].get_activation() for p in range(len(self.layers[l - 1]))])
                if self.biases:
                    net += self.biases[l - 1]
                self.layers[l][n].set_input(net)
        
    def backpropagate(self, key, eta):
        for a in range(len(key)):
            self.layers[-1][a].evaluate(key[a])
        #output_error = sum([n.get_error() for n in self.layers[-1] ])
        new_weights = [[[0 for w in range(len(self.weights[m][n]))] for n in range(len(self.weights[m]))] for m in range(len(self.weights))]
        for l in range(len(self.layers) - 1, 0, -1):
            for o in range(len(self.layers[l])):
                for i in range(len(self.layers[l - 1])):
                    if l == len(self.layers) - 1:
                        f = self.layers[l][o].get_activation() - key[o]
                    else:
                        node_effect = sum([self.weights[l][n][o] * self.layers[l + 1][n].get_effect() for n in range(len(self.layers[l + 1]))])
                        self.layers[l][o].set_effect(node_effect)
                        f = sum([(node_effect * (self.layers[l + 1][n].get_derivative())) * self.weights[l][n][o] for n in range(len(self.layers[l + 1]))])
                    s = self.layers[l][o].get_derivative()
                    t = self.layers[l - 1][i].get_activation()
                    delta = f * s * t
                    new_weights[l - 1][o][i] = self.weights[l - 1][o][i] - eta * delta
        self.weights = new_weights
        
    def train(self, inputs, answers, eta=0.5, epoches=1):
        if len(inputs) != len(answers):
            raise ValueError("The number of input arrays and answer arrays must match!")
        for i in range(len(inputs)):
            if len(inputs[i]) != len(self.layers[0]):
                raise ValueError("Each input array must be the same size as the called network's input layer width!")
        for i in range(len(answers)):
            if len(answers[i]) != len(self.layers[-1]):
                raise ValueError("Each answer array must be the same size as the called network's output layer width!")
        if len(inputs) == 1:
            iterations = epoches
        else:
            iterations = 1
        for _ in range(epoches):
            for i in range(len(inputs)):
                for n in range(len(self.layers[0])):
                    self.layers[0][n].set_activation(inputs[i][n])
                for __ in range(iterations):
                    self.feedforward()
                    self.backpropagate(answers[i], eta)
    
    def test(self, inputs):
        for i in range(len(inputs)):
            if len(inputs[i]) != len(self.layers[0]):
                print("Skipping input array " + str(i + 1) +  " as the number of inputs given does not match the the number of nodes in the called network's input layer!")
                continue
            for n in range(len(self.layers[0])):
                self.layers[0][n].set_activation(inputs[i][n])
            self.feedforward()
            self.print_inputs()
            self.print_outputs()
            

# for testing
iris_dataset = [[5.1, 3.5, 1.4, 0.2, 'I. setosa'],[4.9, 3.0, 1.4, 0.2, 'I. setosa'],[4.7, 3.2, 1.3, 0.2, 'I. setosa'],[4.6, 3.1, 1.5, 0.2, 'I. setosa'],[5.0, 3.6, 1.4, 0.3, 'I. setosa'],[5.4, 3.9, 1.7, 0.4, 'I. setosa'],[4.6, 3.4, 1.4, 0.3, 'I. setosa'],[5.0, 3.4, 1.5, 0.2, 'I. setosa'],[4.4, 2.9, 1.4, 0.2, 'I. setosa'],[4.9, 3.1, 1.5, 0.1, 'I. setosa'],[5.4, 3.7, 1.5, 0.2, 'I. setosa'],[4.8, 3.4, 1.6, 0.2, 'I. setosa'],[4.8, 3.0, 1.4, 0.1, 'I. setosa'],[4.3, 3.0, 1.1, 0.1, 'I. setosa'],[5.8, 4.0, 1.2, 0.2, 'I. setosa'],[5.7, 4.4, 1.5, 0.4, 'I. setosa'],[5.4, 3.9, 1.3, 0.4, 'I. setosa'],[5.1, 3.5, 1.4, 0.3, 'I. setosa'],[5.7, 3.8, 1.7, 0.3, 'I. setosa'],[5.1, 3.8, 1.5, 0.3, 'I. setosa'],[5.4, 3.4, 1.7, 0.2, 'I. setosa'],[5.1, 3.7, 1.5, 0.4, 'I. setosa'],[4.6, 3.6, 1.0, 0.2, 'I. setosa'],[5.1, 3.3, 1.7, 0.5, 'I. setosa'],[4.8, 3.4, 1.9, 0.2, 'I. setosa'],[5.0, 3.0, 1.6, 0.2, 'I. setosa'],[5.0, 3.4, 1.6, 0.4, 'I. setosa'],[5.2, 3.5, 1.5, 0.2, 'I. setosa'],[5.2, 3.4, 1.4, 0.2, 'I. setosa'],[4.7, 3.2, 1.6, 0.2, 'I. setosa'],[4.8, 3.1, 1.6, 0.2, 'I. setosa'],[5.4, 3.4, 1.5, 0.4, 'I. setosa'],[5.2, 4.1, 1.5, 0.1, 'I. setosa'],[5.5, 4.2, 1.4, 0.2, 'I. setosa'],[4.9, 3.1, 1.5, 0.2, 'I. setosa'],[5.0, 3.2, 1.2, 0.2, 'I. setosa'],[5.5, 3.5, 1.3, 0.2, 'I. setosa'],[4.9, 3.6, 1.4, 0.1, 'I. setosa'],[4.4, 3.0, 1.3, 0.2, 'I. setosa'],[5.1, 3.4, 1.5, 0.2, 'I. setosa'],[5.0, 3.5, 1.3, 0.3, 'I. setosa'],[4.5, 2.3, 1.3, 0.3, 'I. setosa'],[4.4, 3.2, 1.3, 0.2, 'I. setosa'],[5.0, 3.5, 1.6, 0.6, 'I. setosa'],[5.1, 3.8, 1.9, 0.4, 'I. setosa'],[4.8, 3.0, 1.4, 0.3, 'I. setosa'],[5.1, 3.8, 1.6, 0.2, 'I. setosa'],[4.6, 3.2, 1.4, 0.2, 'I. setosa'],[5.3, 3.7, 1.5, 0.2, 'I. setosa'],[5.0, 3.3, 1.4, 0.2, 'I. setosa'],[7.0, 3.2, 4.7, 1.4, 'I. versicolor'],[6.4, 3.2, 4.5, 1.5, 'I. versicolor'],[6.9, 3.1, 4.9, 1.5, 'I. versicolor'],[5.5, 2.3, 4.0, 1.3, 'I. versicolor'],[6.5, 2.8, 4.6, 1.5, 'I. versicolor'],[5.7, 2.8, 4.5, 1.3, 'I. versicolor'],[6.3, 3.3, 4.7, 1.6, 'I. versicolor'],[4.9, 2.4, 3.3, 1.0, 'I. versicolor'],[6.6, 2.9, 4.6, 1.3, 'I. versicolor'],[5.2, 2.7, 3.9, 1.4, 'I. versicolor'],[5.0, 2.0, 3.5, 1.0, 'I. versicolor'],[5.9, 3.0, 4.2, 1.5, 'I. versicolor'],[6.0, 2.2, 4.0, 1.0, 'I. versicolor'],[6.1, 2.9, 4.7, 1.4, 'I. versicolor'],[5.6, 2.9, 3.6, 1.3, 'I. versicolor'],[6.7, 3.1, 4.4, 1.4, 'I. versicolor'],[5.6, 3.0, 4.5, 1.5, 'I. versicolor'],[5.8, 2.7, 4.1, 1.0, 'I. versicolor'],[6.2, 2.2, 4.5, 1.5, 'I. versicolor'],[5.6, 2.5, 3.9, 1.1, 'I. versicolor'],[5.9, 3.2, 4.8, 1.8, 'I. versicolor'],[6.1, 2.8, 4.0, 1.3, 'I. versicolor'],[6.3, 2.5, 4.9, 1.5, 'I. versicolor'],[6.1, 2.8, 4.7, 1.2, 'I. versicolor'],[6.4, 2.9, 4.3, 1.3, 'I. versicolor'],[6.6, 3.0, 4.4, 1.4, 'I. versicolor'],[6.8, 2.8, 4.8, 1.4, 'I. versicolor'],[6.7, 3.0, 5.0, 1.7, 'I. versicolor'],[6.0, 2.9, 4.5, 1.5, 'I. versicolor'],[5.7, 2.6, 3.5, 1.0, 'I. versicolor'],[5.5, 2.4, 3.8, 1.1, 'I. versicolor'],[5.5, 2.4, 3.7, 1.0, 'I. versicolor'],[5.8, 2.7, 3.9, 1.2, 'I. versicolor'],[6.0, 2.7, 5.1, 1.6, 'I. versicolor'],[5.4, 3.0, 4.5, 1.5, 'I. versicolor'],[6.0, 3.4, 4.5, 1.6, 'I. versicolor'],[6.7, 3.1, 4.7, 1.5, 'I. versicolor'],[6.3, 2.3, 4.4, 1.3, 'I. versicolor'],[5.6, 3.0, 4.1, 1.3, 'I. versicolor'],[5.5, 2.5, 4.0, 1.3, 'I. versicolor'],[5.5, 2.6, 4.4, 1.2, 'I. versicolor'],[6.1, 3.0, 4.6, 1.4, 'I. versicolor'],[5.8, 2.6, 4.0, 1.2, 'I. versicolor'],[5.0, 2.3, 3.3, 1.0, 'I. versicolor'],[5.6, 2.7, 4.2, 1.3, 'I. versicolor'],[5.7, 3.0, 4.2, 1.2, 'I. versicolor'],[5.7, 2.9, 4.2, 1.3, 'I. versicolor'],[6.2, 2.9, 4.3, 1.3, 'I. versicolor'],[5.1, 2.5, 3.0, 1.1, 'I. versicolor'],[5.7, 2.8, 4.1, 1.3, 'I. versicolor'],[6.3, 3.3, 6.0, 2.5, 'I. virginica'],[5.8, 2.7, 5.1, 1.9, 'I. virginica'],[7.1, 3.0, 5.9, 2.1, 'I. virginica'],[6.3, 2.9, 5.6, 1.8, 'I. virginica'],[6.5, 3.0, 5.8, 2.2, 'I. virginica'],[7.6, 3.0, 6.6, 2.1, 'I. virginica'],[4.9, 2.5, 4.5, 1.7, 'I. virginica'],[7.3, 2.9, 6.3, 1.8, 'I. virginica'],[6.7, 2.5, 5.8, 1.8, 'I. virginica'],[7.2, 3.6, 6.1, 2.5, 'I. virginica'],[6.5, 3.2, 5.1, 2.0, 'I. virginica'],[6.4, 2.7, 5.3, 1.9, 'I. virginica'],[6.8, 3.0, 5.5, 2.1, 'I. virginica'],[5.7, 2.5, 5.0, 2.0, 'I. virginica'],[5.8, 2.8, 5.1, 2.4, 'I. virginica'],[6.4, 3.2, 5.3, 2.3, 'I. virginica'],[6.5, 3.0, 5.5, 1.8, 'I. virginica'],[7.7, 3.8, 6.7, 2.2, 'I. virginica'],[7.7, 2.6, 6.9, 2.3, 'I. virginica'],[6.0, 2.2, 5.0, 1.5, 'I. virginica'],[6.9, 3.2, 5.7, 2.3, 'I. virginica'],[5.6, 2.8, 4.9, 2.0, 'I. virginica'],[7.7, 2.8, 6.7, 2.0, 'I. virginica'],[6.3, 2.7, 4.9, 1.8, 'I. virginica'],[6.7, 3.3, 5.7, 2.1, 'I. virginica'],[7.2, 3.2, 6.0, 1.8, 'I. virginica'],[6.2, 2.8, 4.8, 1.8, 'I. virginica'],[6.1, 3.0, 4.9, 1.8, 'I. virginica'],[6.4, 2.8, 5.6, 2.1, 'I. virginica'],[7.2, 3.0, 5.8, 1.6, 'I. virginica'],[7.4, 2.8, 6.1, 1.9, 'I. virginica'],[7.9, 3.8, 6.4, 2.0, 'I. virginica'],[6.4, 2.8, 5.6, 2.2, 'I. virginica'],[6.3, 2.8, 5.1, 1.5, 'I. virginica'],[6.1, 2.6, 5.6, 1.4, 'I. virginica'],[7.7, 3.0, 6.1, 2.3, 'I. virginica'],[6.3, 3.4, 5.6, 2.4, 'I. virginica'],[6.4, 3.1, 5.5, 1.8, 'I. virginica'],[6.0, 3.0, 4.8, 1.8, 'I. virginica'],[6.9, 3.1, 5.4, 2.1, 'I. virginica'],[6.7, 3.1, 5.6, 2.4, 'I. virginica'],[6.9, 3.1, 5.1, 2.3, 'I. virginica'],[5.8, 2.7, 5.1, 1.9, 'I. virginica'],[6.8, 3.2, 5.9, 2.3, 'I. virginica'],[6.7, 3.3, 5.7, 2.5, 'I. virginica'],[6.7, 3.0, 5.2, 2.3, 'I. virginica'],[6.3, 2.5, 5.0, 1.9, 'I. virginica'],[6.5, 3.0, 5.2, 2.0, 'I. virginica'],[6.2, 3.4, 5.4, 2.3, 'I. virginica'],[5.9, 3.0, 5.1, 1.8, 'I. virginica']]
random.shuffle(iris_dataset)
iris_training_set = iris_dataset[:int(len(iris_dataset) * 0.8)]
iris_test_set = iris_dataset[int(len(iris_dataset) * 0.8):]

