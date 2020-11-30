from Neuron import Neuron

class Layer:
    def __init__(self, name, num_inputs, num_neurons):
        self.name = name
        self.neurons = [Neuron(num_inputs) for i in range(num_neurons)]

    # Update weights given inputs and learning rate.
    def update_weights(self, inputs, learning_rate, alpha):
        for neuron in self.neurons:
            neuron.bias += (learning_rate * neuron.delta) + (alpha*neuron.prev_delta)
            for i in range(len(inputs)):
                neuron.weights[i+1] += learning_rate * neuron.delta * inputs[i] + (alpha * neuron.prev_delta)
            neuron.prev_delta = neuron.delta

    # Returns all the neuron outputs of the layer.
    def get_outputs(self):
        return [neuron.output for neuron in self.neurons]

    # Activate all neurons in the layer and 
    def activate(self, inputs):
        layer_outputs = []
        for neuron in self.neurons:
            neuron.activate(inputs)
            layer_outputs.append(neuron.output)
        return layer_outputs

    def calc_output_deltas(self, expected):
        for i in range(len(self.neurons)):
            neuron = self.neurons[i]
            neuron.delta = (expected[i] - neuron.output) * neuron.derivative()
        return

    def calc_hidden_deltas(self, output_deltas):
        for curr_index, curr_neuron in enumerate(self.neurons):
            error = 0.0
            for output_neuron in output_deltas.neurons:
                error += (output_neuron.weights[curr_index+1] * output_neuron.delta)
            curr_neuron.delta = error * curr_neuron.derivative()

    def describe(self):
        print("Layer: ", self.name)
        for index, neuron in enumerate(self.neurons):
            print("Neuron: ", index + 1)
            print("Bias: ", neuron.bias)
            print("Weights: ", neuron.weights)
        return