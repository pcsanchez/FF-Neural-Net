from random import random
from math import exp

class Neuron:
    def __init__(self, num_inputs):
        self.weights = {i+1: random() for i in range(num_inputs)}
        self.bias = random()
        self.output = -1
        self.delta = -1
        self.prev_changes = [0 for i in range(num_inputs+1)]

    def __output(self, inputs):
        weighted_sum = self.bias
        w = list(self.weights.values())
        for i in range(len(inputs)):
            weighted_sum += w[i] * inputs[i]
        return weighted_sum

    def activate(self, inputs):
        exponent = self.__output(inputs)
        if exponent > 200:
            self.output = 1
        elif exponent < -200:
            self.output = 0
        else:
            self.output =  1.0 / (1.0 + exp(-self.__output(inputs)))
        return

    def derivative(self):
        return self.output * (1.0 - self.output)
            
