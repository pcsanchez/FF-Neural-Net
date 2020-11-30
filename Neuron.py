from random import random
from math import exp

class Neuron:
    def __init__(self, num_inputs):
        self.weights = {i+1: random() for i in range(num_inputs)}
        self.bias = random()
        self.output = -1
        self.delta = -1
        self.prev_delta = 0

    def __output(self, inputs):
        weighted_sum = self.bias
        w = list(self.weights.values())
        for i in range(len(inputs)):
            weighted_sum += w[i] * inputs[i]
        return weighted_sum

    def activate(self, inputs):
        self.output =  1.0 / (1.0 + exp(-self.__output(inputs)))
        return

    def derivative(self):
        return self.output * (1.0 - self.output)
            
