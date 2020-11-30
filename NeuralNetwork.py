from Layer import Layer

class NeuralNetwork:
    def __init__(self, num_inputs, num_hidden_neurons, num_outputs):
        hidden_layer = Layer("Hidden Layer", num_inputs,num_hidden_neurons)
        output_layer = Layer("Output Layer", num_hidden_neurons, num_outputs)
        self.layers = [hidden_layer, output_layer]

    def describe(self):
        print('Neural Network Description')
        print('=========================================================')
        for layer in self.layers:
            layer.describe()
            print('=========================================================')
        return

    def __forward_propagate(self,input):
        inputs = input
        for layer in self.layers:
            inputs = layer.activate(inputs)
        return inputs

    def __back_propagate(self, expected):
        hidden_layer = self.layers[0]
        output_layer = self.layers[1]
        output_layer.calc_output_deltas(expected)
        hidden_layer.calc_hidden_deltas(output_layer)

    def __update_weights(self, inputs):
        hidden_layer = self.layers[0]
        output_layer = self.layers[1]
        hidden_outputs = hidden_layer.get_outputs()
        output_layer.update_weights(hidden_outputs, self.learning_rate, self.alpha)
        hidden_layer.update_weights(inputs, self.learning_rate, self.alpha)

    def set_params(self, learning_rate, alpha):
        self.learning_rate = learning_rate
        self.alpha = alpha

    def train(self, epochs, X, y):
        for epoch in range(epochs):
            epoch_error = 0.0
            for index, input in enumerate(X):
                outputs = self.__forward_propagate(input)
                epoch_error += sum([(y[index][i] - outputs[i])**2 for i in range(len(outputs))])
                self.__back_propagate(y[index])
                self.__update_weights(y[index])
            epoch_error = epoch_error / len(X)
            print("Epoch: %d, Training Error: %.3f" % (epoch+1, epoch_error))
        

    def predict(self, input):
        return self.__forward_propagate(input)